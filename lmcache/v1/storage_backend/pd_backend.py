# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
import math
import threading

# Third Party
from mooncake.store import (
    MooncakeDistributedStore,
    ReplicateConfig,
    bind_to_numa_node,
)
import msgspec
import torch
import zmq

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)
from lmcache.v1.system_detection import NUMADetector

logger = init_logger(__name__)


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages."""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message."""

    keys: list[str]
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int


class AllocResponse(PDMsgBase):
    """Allocation response message."""

    already_sent_indexes: list[int]
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


@dataclass
class PDConfig:
    role: str
    proxy_host: Optional[str]
    proxy_port: Optional[int]
    buffer_size: int
    buffer_device: str

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        return PDConfig(
            role=config.pd_role,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=config.pd_buffer_device,
        )


@dataclass
class PendingChunk:
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat
    mem_obj: Optional[MemoryObj] = None
    ready: bool = False

    @property
    def num_bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


class PDBackend(AllocatorBackendInterface):
    """PD backend that stages KV chunks via Mooncake distributed store."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        self.running = True
        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.mooncake_config = MooncakeStoreConfig.load_from_lmcache_config(config)
        self.mooncake_store = MooncakeDistributedStore()
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1
        self.registered_gpu_ptr: Optional[int] = None

        self._setup_mooncake(metadata, config)

        self.zmq_context: Optional[zmq.Context] = None
        self.proxy_side_channel = None

        if self.pd_config.role == "sender":
            self.zmq_context = get_zmq_context(use_asyncio=False)
            self._init_sender()

        self.full_chunk_size = config.chunk_size
        self.meta_shape = torch.Size(metadata.kv_shape)
        self.meta_dtype = metadata.kv_dtype
        self.meta_fmt = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        # First Party
        from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

        corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )

        logger.info(f"Setting cuda device to {corrected_device}")
        torch.cuda.set_device(corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        paged_mem_allocator.init_gpu_memory_allocator(
            config.pd_buffer_size,
            torch.Size(metadata.kv_shape),
            metadata.kv_dtype,
            MemoryFormat.KV_2LTD,
            corrected_device,
        )

        return paged_mem_allocator

    def _setup_mooncake(
        self,
        metadata: LMCacheEngineMetadata,
        config: LMCacheEngineConfig,
    ) -> None:
        numa_mapping = NUMADetector.get_numa_mapping(config)
        if numa_mapping:
            current_device_id = torch.cuda.current_device()
            gpu_to_numa = getattr(numa_mapping, "gpu_to_numa_mapping", {})
            numa_id = gpu_to_numa.get(current_device_id)
            if numa_id is not None:
                bind_to_numa_node(numa_id)
                logger.info(
                    "Mooncake bind_to_numa_node success for GPU %s -> NUMA %s",
                    current_device_id,
                    numa_id,
                )

        setup_ret = self.mooncake_store.setup(
            self.mooncake_config.local_hostname,
            self.mooncake_config.metadata_server,
            self.mooncake_config.global_segment_size,
            self.mooncake_config.local_buffer_size,
            self.mooncake_config.protocol,
            self.mooncake_config.device_name,
            self.mooncake_config.master_server_address,
        )
        if setup_ret != 0:
            raise RuntimeError(
                "Mooncake store setup failed with error code %s" % setup_ret
            )
        logger.info(
            "Mooncake store setup succeeded with config %s", self.mooncake_config
        )

        if self.mooncake_config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.mooncake_store.get_hostname()

        gpu_allocator = self.memory_allocator.gpu_allocator
        ptr = gpu_allocator.buffer_ptr
        size = gpu_allocator.buffer_size
        result = self.mooncake_store.register_buffer(ptr, size)
        if result != 0:
            raise RuntimeError(
                "Mooncake failed to register GPU buffer ptr=%s size=%s err=%s"
                % (hex(ptr), size, result)
            )
        self.registered_gpu_ptr = ptr
        logger.info("Mooncake registered GPU buffer: ptr=%s size=%s", hex(ptr), size)

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        return self.memory_allocator

    def get_allocator_backend(self):
        return self

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        return self.memory_allocator.allocate(
            shape=shape, dtype=dtype, fmt=fmt, allocator_type="gpu"
        )

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        return self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, fmt, allocator_type="gpu"
        )

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return self.mooncake_store.is_exist(key.to_string())

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def _init_sender(self):
        if self.pd_config.proxy_host is None or self.pd_config.proxy_port is None:
            raise ValueError(
                "PD sender requires pd_proxy_host and pd_proxy_port to be configured"
            )
        if self.zmq_context is None:
            raise RuntimeError("ZMQ context must be initialized for PD sender")
        proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
        self.proxy_side_channel = get_zmq_socket(
            self.zmq_context,
            proxy_url,
            "tcp",
            zmq.PUSH,
            "connect",
        )

    def _batch_put_to_mooncake(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        if not keys:
            return

        key_strs = [k.to_string() for k in keys]
        buffer_ptrs = [obj.data_ptr for obj in memory_objs]
        buffer_sizes = [obj.get_size() for obj in memory_objs]

        put_results = self.mooncake_store.batch_put_from(
            key_strs,
            buffer_ptrs,
            buffer_sizes,
            self.replica_config,
        )

        for idx, ret in enumerate(put_results):
            if ret != 0:
                raise RuntimeError(
                    "Mooncake batch_put_from failed for %s with code %s"
                    % (key_strs[idx], ret)
                )
        for obj in memory_objs:
            obj.ref_count_down()

    def _batched_submit_put_task_impl(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        self._batch_put_to_mooncake(keys, memory_objs)
        if (
            transfer_spec
            and getattr(transfer_spec, "is_last_prefill", False)
            and self.proxy_side_channel is not None
        ):
            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            self.proxy_side_channel.send(notif_msg_bytes)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        # create a thread to do this
        threading.Thread(
            target=self._batched_submit_put_task_impl,
            args=(keys, memory_objs, transfer_spec),
            daemon=True,
        ).start()

    def _mooncake_exists(self, key_strs: Sequence[str]) -> list[bool]:
        if isinstance(key_strs, str):  # guard against accidental str iteration
            key_list = [key_strs]
        else:
            key_list = list(key_strs)

        if not key_list:
            return []

        rets = self.mooncake_store.batch_is_exist(key_list)
        return [ret == 1 for ret in rets]

    def reshape_partial_chunk(
        self,
        memory_obj: MemoryObj,
        bytes_read: int,
    ) -> MemoryObj:
        """Trim `memory_obj` to match the actual bytes read from remote storage."""

        dtype = memory_obj.meta.dtype
        if dtype is None:
            raise ValueError(
                "memory_obj meta dtype is required to reshape partial chunk"
            )

        fmt = memory_obj.meta.fmt
        shape_list = list(memory_obj.meta.shape)
        token_dim = fmt.token_dim()

        elements_per_token = 1
        for dim_idx, dim_size in enumerate(shape_list):
            if dim_idx == token_dim:
                continue
            elements_per_token *= dim_size

        dtype_size = torch.tensor([], dtype=dtype).element_size()
        single_token_size = elements_per_token * dtype_size
        full_chunk_size = single_token_size * shape_list[token_dim]

        if bytes_read % single_token_size != 0 or bytes_read > full_chunk_size:
            raise ValueError(
                f"bytes_read: {bytes_read} is illegal, "
                f"single_token_size: {single_token_size}, "
                f"full_chunk_size: {full_chunk_size}"
            )

        if bytes_read == full_chunk_size:
            return memory_obj

        actual_tokens = bytes_read // single_token_size
        shape_list[token_dim] = actual_tokens
        memory_obj.raw_data = memory_obj.raw_data[:bytes_read]
        memory_obj.meta.shape = torch.Size(shape_list)

        return memory_obj

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        raise NotImplementedError("PDBackend put is not implemented")

    def batched_get_blocking_impl(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        mem_objs: list[Optional[MemoryObj]] = []
        valid_indices: list[int] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        for idx, key in enumerate(keys):
            mem_obj = self.allocate(self.meta_shape, self.meta_dtype, self.meta_fmt)
            mem_objs.append(mem_obj)
            if mem_obj is None:
                logger.error(
                    "PDBackend sender failed to allocate buffer for key %s",
                    key.to_string(),
                )
                continue

            valid_indices.append(idx)
            buffer_ptrs.append(mem_obj.data_ptr)
            buffer_sizes.append(mem_obj.get_size())

        if valid_indices:
            key_strs = [keys[idx].to_string() for idx in valid_indices]
            bytes_read = self.mooncake_store.batch_get_into(
                key_strs, buffer_ptrs, buffer_sizes
            )
            if len(bytes_read) != len(valid_indices):
                raise RuntimeError(
                    "Mooncake batch_get_into returned %s entries for %s keys"
                    % (len(bytes_read), len(valid_indices))
                )

            for offset, num_bytes in enumerate(bytes_read):
                idx = valid_indices[offset]
                mem_obj = mem_objs[idx]
                if mem_obj is None:
                    logger.error(
                        "Unexpected None MemoryObj during reshape for key %s",
                        keys[idx].to_string(),
                    )
                    continue
                self.reshape_partial_chunk(mem_obj, num_bytes)

        return mem_objs

    def batched_get_blocking(
        self, keys: list[CacheEngineKey]
    ) -> list[Optional[MemoryObj]]:
        return self.batched_get_blocking_impl(keys)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        raise NotImplementedError("PDBackend get_blocking is not implemented")

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        raise NotImplementedError(
            "PDBackend batched_get_non_blocking is not implemented"
        )

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        return True

    def close(self) -> None:
        self.running = False
        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None
        self.proxy_side_channel = None

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True
