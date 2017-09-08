proto_shmem
===========
* **What**: a prototype implementation of shared memory segments for use in Python's multiprocessing module.
* **Why**: to provide a path for better parallel execution performance by eliminating the need to transmit data between distinct processes on a single system (not for use in distributed memory architectures).


POSIX/SystemV/Native Shared Memory
----------------------------------
* tools like posix_ipc look very promising for Linux/BSD variants but not for Windows
* PostgreSQL has a consistent internal API for offering shared memory across Windows/Unix platforms
    * based on System V, made NetBSD/OpenBSD happy before they supported POSIX
    * not immediately extractable but could be done
* "shared-array", "shared_ndarray", and "sharedmem-numpy" all have interesting implementations for exposing NumPy arrays via shared memory segments

Design
------
* use of Manager is required to enforce access rules in different processes via proxy objects
* use of context is all that's required for memory management (ensures free-ing) when access rules unneeded

* existing sharedctypes submodule uses a single shared memory segment through the heap submodule with its own memory management implementation to "malloc" space for allowed ctypes and then "free" that space when no longer used, recycling it for use again from the shared memory segment

Use Cases
---------
* Want #1:
    ```
    #### in process alpha (same device)
    shm_a = mp.attach_or_create_shmem("unique_id_001")  # likely enforce length constraint on name, combine into __init__ args
    m = shm_a.list([4, 5, 6, 7])
    print(m[0])  # prints 4, access through shared memory segment, no serialization
    mgr = mp.Manager(address=('localhost', 6001), authkey=b'friend')
    mgr.register("get_shared_list_m", callable=lambda: m)

    #### in process beta (same device)
    shm_b = remote_mgr.attach_or_create_shmem("unique_id_001")
    mp.Manager.register("get_shared_list_m")
    remote_mgr = mp.Manager(address=('localhost', 6001), authkey=b'friend')
    remote_mgr.connect()
    m = remote_mgr.get_shared_list_m()  # creation of proxy object depends on shm_b existing... open question how to hook
    print(m[0])  # prints 4, access through shared memory segment
    m[0] = -1000  # serializes __setitem__ call and sends to manager to perform actual mutation
    print(m[0])  # prints -1000, access through shared memory segment

    #### back in process alpha again
    print(m[0])  # prints -1000, access through shared memory segment
    ```

* Want #2:
    ```
    shm = mp.attach_or_create_shmem("unique_id_002")
    m = shm.list([4, 5, 6, 7])
    with mp.Pool(processes=4) as p:
        _results = p.map(compute_intensive_func, m)  # Wish: all accesses through shared memory segment, no serialization
        results = p.map(compute_intensive_func_2, enumerate([m] * 4))  # inside accesses m[i] through shared memory segment
    print(results)
    ```

* Want (note the pseudocode in those lambdas) #3:
    ```
    shm = mp.attach_or_create_shmem("unique_id_003")
    local_r = np.random.random_sample((4000,))
    shared_r = shm.ndarray(local_r)  # probably need to provide mechanism to register this to support numpy
    with mp.Pool(processes=4) as p:
        _results = p.map(lambda i: shared_r[i::4] = shared_r[i::4] ** 0.5, range(4))  # all through shared memory
        _results = p.map(lambda i: shared_r[1000*i:1000*(i-1)] = shared_r[1000*i:1000*(i-1)] * 4, range(4))  # all shared memory
    print(shared_r)  # access through shared memory
    ```

TODO
----
* expose a SharedDict class (like the now existing SharedList)
* evaluate continued use of posix_ipc as a whole or only parts relevant to shared memory segments (not using POSIX message queues or POSIX semaphores at present) or inspired new implementation
* eliminate the explicit dependency on NumPy's ndarrays but keep it dead simple to create ndarrays in shared memory segments
* rationalize all this with what's done in multiprocessing.sharedctypes
    - directly compare performance of sharedctypes's all-in-one segment approach versus multiple segments approach used here so far
* demo performance differences between using shared memory segments versus not with multiprocessing
