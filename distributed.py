import threading
import time

# --- SHARED RESOURCES ---
database = 0
lock = threading.Lock()      # Mutex: Only 1 at a time
sem = threading.Semaphore(2) # Semaphore: Max 2 at a time

# --- DEMO 1: MUTEX (Data Safety) ---
def safe_writer(thread_id):
    global database
    with lock:  # <--- CRITICAL SECTION STARTS
        print(f"[Mutex] Thread {thread_id} has the lock. Updating...")
        local_copy = database
        time.sleep(0.2)  # Simulate slow writing
        database = local_copy + 1
        # <--- CRITICAL SECTION ENDS (Lock released)

def unsafe_writer(thread_id):
    global database
    print(f"[Mutex] Thread {thread_id} has the lock. Updating...")
    local_copy = database
    time.sleep(0.2)  # Simulate slow writing
    database = local_copy + 1

# --- DEMO 2: SEMAPHORE (Rate Limiting) ---
def limited_api_caller(thread_id):
    print(f"[Sem] Thread {thread_id} waiting in line...")
    with sem:   # <--- WAITS HERE if 2 threads are already inside
        print(f"--> [Sem] Thread {thread_id} calling API! (Active)")
        time.sleep(1.0) # Simulate 1 second API call
        print(f"<-- [Sem] Thread {thread_id} finished.")

# ==========================================
# ACTUAL EXECUTION
# ==========================================

# print("--- PART 1: MUTEX DEMO (Sequential) ---")
# # Creates 5 threads that want to update the database
# writer_threads = [threading.Thread(target=safe_writer, args=(i,)) for i in range(5)]

# for t in writer_threads: t.start()
# for t in writer_threads: t.join()

# print(f"Final Database Value: {database}\n")


# print("--- PART 2: SEMAPHORE DEMO (Concurrency) ---")
# # Creates 6 threads, but Semaphore only allows 2 at a time
# api_threads = [threading.Thread(target=limited_api_caller, args=(i,)) for i in range(6)]

# for t in api_threads: t.start()
# for t in api_threads: t.join()

print("--- PART 3: Unsafe demo ---")
# Creates 5 threads that want to update the database
writer_threads = [threading.Thread(target=unsafe_writer, args=(i,)) for i in range(5)]

for t in writer_threads: t.start()
for t in writer_threads: t.join()
print(f"Final Database Value: {database}\n")
