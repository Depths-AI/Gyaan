import requests
import time
import multiprocessing
import random
import os
import polars as pl
import numpy as np
from statistics import mean, median
from multiprocessing import Queue
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
import threading
from shutil import rmtree
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.utils.io import create_table, insert_table, read_table


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('insertion_concurrency_test')

# Constants
NUM_PROCESSES = 10  # Number of concurrent processes
NUM_BATCHES = 1     # Number of batches per process
ROWS_PER_BATCH = 100  # Number of rows per batch
BASE_URL = "http://127.0.0.1:8000"
DELTA_TABLE_PATH = "./test_delta_table"  # Path to store the Delta table
DELTA_URI = f"file://{os.path.abspath(DELTA_TABLE_PATH)}"  # Full URI for Delta table

# FastAPI server setup
app = FastAPI()
server_running = False
server_instance = None

# API endpoints for Delta operations
@app.post("/create-table")
async def create_table_endpoint(background_tasks: BackgroundTasks):
    """Create a new Delta table with initial data"""
    try:
        # Create initial dataframe with one row
        initial_df = pl.DataFrame({
            "id": [0],
            "value": [0.0],
            "process": [0],
            "batch": [0],
            "timestamp": [time.time()]
        })
        
        # Create the table
        background_tasks.add_task(create_table, DELTA_URI, initial_df, mode="overwrite")
        return {"status": "success", "message": "Table creation initiated"}
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create table: {str(e)}")


@app.post("/insert-data")
async def insert_data_endpoint(data: dict, background_tasks: BackgroundTasks):
    """Insert data into the Delta table"""
    try:
        # Convert the data to a polars DataFrame
        df = pl.DataFrame(data)
        
        # Insert the data
        background_tasks.add_task(insert_table, DELTA_URI, df)
        return {"status": "success", "message": "Data insertion initiated"}
    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert data: {str(e)}")


@app.get("/read-table")
async def read_table_endpoint():
    """Read data from the Delta table"""
    try:
        # Read the table
        df = await read_table(DELTA_URI)
        
        # Convert to dict for JSON response
        result = {col: df[col].to_list() for col in df.columns}
        return result
    except Exception as e:
        logger.error(f"Error reading table: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read table: {str(e)}")


@app.get("/table-stats")
async def table_stats_endpoint():
    """Get statistics about the Delta table"""
    try:
        df = await read_table(DELTA_URI)
        
        # Create process count data in a JSON-serializable format
        process_counts = df.group_by("process").agg(pl.len()).sort("process")
        process_counts_dict = {
            "process": process_counts["process"].to_list(),
            "count": process_counts["len"].to_list()
        }
        
        stats = {
            "total_rows": df.height,
            "unique_processes": df["process"].n_unique(),
            "rows_per_process": process_counts_dict
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting table stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get table stats: {str(e)}")


# Server functions
def run_server():
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


def start_server():
    """Start the server in a separate thread"""
    global server_running, server_instance
    
    if server_running:
        return
    
    # Start the server in a separate thread
    server_instance = threading.Thread(target=run_server)
    server_instance.daemon = True
    server_instance.start()
    server_running = True
    
    # Give the server a moment to start
    time.sleep(2)
    logger.info("Server started")


def stop_server():
    """Stop the server"""
    global server_running, server_instance
    
    if not server_running:
        return
    
    try:
        # Make a request to shut down the server
        requests.get(f"{BASE_URL}/docs")
    except:
        pass
    
    server_running = False
    logger.info("Server shutdown initiated")


# Client simulator function
def simulate_client(client_id, num_batches, rows_per_batch, result_queue):
    """Simulate a client making multiple batch insertion requests"""
    session = requests.Session()
    results = []
    
    for batch in range(num_batches):
        # Create data for this batch
        data = {
            "id": [client_id * num_batches * rows_per_batch + batch * rows_per_batch + i 
                  for i in range(rows_per_batch)],
            "value": [random.random() for _ in range(rows_per_batch)],
            "process": [client_id] * rows_per_batch,
            "batch": [batch] * rows_per_batch,
            "timestamp": [time.time()] * rows_per_batch
        }
        
        # Add small random delay to simulate real-world behavior
        time.sleep(random.uniform(0.05, 0.2))
        
        # Send the insertion request
        start_time = time.time()
        try:
            response = session.post(f"{BASE_URL}/insert-data", json=data)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = {
                    "success": True,
                    "client_id": client_id,
                    "batch": batch,
                    "rows": rows_per_batch,
                    "latency": latency
                }
            else:
                logger.error(f"Client {client_id}: Error inserting batch {batch}: {response.text}")
                result = {
                    "success": False,
                    "client_id": client_id,
                    "batch": batch,
                    "error": response.text,
                    "latency": latency
                }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Client {client_id}: Exception inserting batch {batch}: {e}")
            result = {
                "success": False,
                "client_id": client_id,
                "batch": batch,
                "error": str(e),
                "latency": latency
            }
        
        results.append(result)
    
    # Put all results from this client in the queue
    result_queue.put(results)


def create_initial_table() -> bool:
    """Create a new Delta table for testing"""
    try:
        response = requests.post(f"{BASE_URL}/create-table")
        if response.status_code == 200:
            logger.info("Delta table created successfully")
            return True
        else:
            logger.error(f"Error creating Delta table: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Exception creating Delta table: {e}")
        return False


def get_table_stats() -> dict:
    """Get statistics about the Delta table"""
    try:
        response = requests.get(f"{BASE_URL}/table-stats")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error getting table stats: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error checking table stats: {e}")
        return None


def cleanup_delta_table():
    """Remove the Delta table directory"""
    try:
        if os.path.exists(DELTA_TABLE_PATH):
            rmtree(DELTA_TABLE_PATH)
            logger.info(f"Successfully removed Delta table directory: {DELTA_TABLE_PATH}")
    except Exception as e:
        logger.error(f"Error cleaning up Delta table: {e}")


def main():
    # Cleanup any existing test data
    cleanup_delta_table()
    
    logger.info("Starting Delta Lake insertion concurrency test")
    
    # Start the FastAPI server
    start_server()
    
    # Create the initial Delta table
    logger.info("Creating initial Delta table...")
    if not create_initial_table():
        logger.error("Failed to create initial Delta table. Aborting test.")
        stop_server()
        return
    
    # Calculate the total expected rows
    expected_total_rows = (NUM_PROCESSES * NUM_BATCHES * ROWS_PER_BATCH) + 1  # +1 for initial row
    logger.info(f"Expected total rows after test: {expected_total_rows}")
    
    # Start the concurrency test
    logger.info(f"\nStarting concurrency test with {NUM_PROCESSES} simultaneous clients")
    logger.info(f"Each client will insert {NUM_BATCHES} batches with {ROWS_PER_BATCH} rows each")
    
    # Create result queue for collecting metrics
    result_queue = Queue()
    
    # Start timing
    start_time = time.time()
    
    # Create and start processes for each client
    processes = []
    for client_id in range(NUM_PROCESSES):
        p = multiprocessing.Process(
            target=simulate_client,
            args=(client_id, NUM_BATCHES, ROWS_PER_BATCH, result_queue)
        )
        processes.append(p)
        p.start()
    
    # Collect all results
    all_results = []
    for _ in range(NUM_PROCESSES):
        client_results = result_queue.get()
        all_results.extend(client_results)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Add a brief delay to allow operations to complete on the server
    logger.info("\nWaiting briefly for server operations to complete...")
    time.sleep(5)
    
    # Calculate success metrics
    successful_requests = [r for r in all_results if r["success"]]
    total_requests = len(all_results)
    latencies = [r["latency"] for r in all_results if r["success"]]
    
    # Get final table statistics
    logger.info("Retrieving final table statistics...")
    stats = get_table_stats()
    
    # Print test results
    logger.info(f"\nTest completed in {total_time:.2f} seconds")
    logger.info(f"Successful requests: {len(successful_requests)} out of {total_requests}")
    logger.info(f"Success rate: {(len(successful_requests) / total_requests) * 100:.1f}%")
    
    # Create timing stats DataFrame
    mean_latency = mean(latencies)
    median_latency = median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    timing_stats = pl.DataFrame({
        "Metric": ["Total Test Time", "Mean Latency", "Median Latency", "Min Latency", "Max Latency", "95th Percentile", "99th Percentile"],
        "Value (seconds)": [total_time, mean_latency, median_latency, min_latency, max_latency, p95_latency, p99_latency],
        "Description": [
            f"{successful_requests} requests in {total_time:.2f}s",
            f"Average request time",
            f"Middle value of all requests",
            f"Fastest request",
            f"Slowest request",
            f"95% of requests faster than this",
            f"99% of requests faster than this"
        ]
    })

    logger.info("\n" + "="*80)
    logger.info("TIMING STATISTICS")
    logger.info("-"*80)
    print("\nTiming Statistics DataFrame:")
    print(timing_stats)
    logger.info("="*80)
    
    logger.info("\nLatency Statistics (seconds):")
    logger.info(f"Mean: {mean(latencies):.3f}")
    logger.info(f"Median: {median(latencies):.3f}")
    logger.info(f"Min: {min(latencies):.3f}")
    logger.info(f"Max: {max(latencies):.3f}")
    logger.info(f"95th percentile: {np.percentile(latencies, 95):.3f}")
    logger.info(f"99th percentile: {np.percentile(latencies, 99):.3f}")
    
    # Verify data integrity
    if stats and "total_rows" in stats:
        final_row_count = stats["total_rows"]
        logger.info(f"\nFinal row count in Delta table: {final_row_count}")
        
        if final_row_count == expected_total_rows:
            logger.info("TEST PASSED: All rows were inserted successfully!")
        else:
            logger.info(f"TEST FAILED: Expected {expected_total_rows} rows, found {final_row_count}")
            
        # Print additional table statistics
        if "unique_processes" in stats:
            logger.info(f"Unique processes: {stats['unique_processes']}")
        
        if "rows_per_process" in stats:
            logger.info("Rows per process:")
            process_data = stats["rows_per_process"]
            for process in sorted(int(p) for p in process_data["process"]):
                idx = process_data["process"].index(process)
                count = process_data["count"][idx]
                logger.info(f"  Process {process}: {count} rows")
    else:
        logger.error("Failed to get final table statistics")
    
    # Read some sample data for verification
    try:
        response = requests.get(f"{BASE_URL}/read-table")
        if response.status_code == 200:
            sample_data = response.json()
            
            # Convert JSON data back to polars DataFrame for proper display
            df = pl.DataFrame({
                "id": sample_data["id"],
                "value": sample_data["value"],
                "process": sample_data["process"],
                "batch": sample_data["batch"],
                "timestamp": sample_data["timestamp"]
            })
            
            logger.info("\n" + "="*80)
            logger.info("FINAL DATAFRAME")
            logger.info("-"*80)
            
            # Print full DataFrame stats
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Memory usage: {df.estimated_size() / 1024:.2f} KB")
            logger.info("-"*80)
            
            # Print actual DataFrame
            print("\nFull DataFrame:")
            print(df)
            print("\nHead (5 rows):")
            print(df.head(5))
            print("\nTail (5 rows):")
            print(df.tail(5))
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print(df.describe())
            
            logger.info("="*80)
    except Exception as e:
        logger.error(f"Error reading sample data: {e}")
    
    # Stop the server and cleanup
    logger.info("\nStopping server...")
    stop_server()
    
    logger.info("Cleaning up test data...")
    cleanup_delta_table()
    
    logger.info("Test complete")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    main()
