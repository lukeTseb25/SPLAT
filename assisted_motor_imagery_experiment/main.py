from modules.motor_imagery_experiment import run_motor_imagery_experiment
from modules.LSLDataCollector import LSLDataCollector
import logging
import threading

def main():
	# Configure logging
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)

	logging.info("Starting Motor Imagery experiment script")
	# Start data collector thread
	stop_event = threading.Event()
	collector = LSLDataCollector(stop_event)
	collector.start()

	# Run the experiment
	run_motor_imagery_experiment()

	# Stop the collector thread gracefully
	logging.info("Experiment complete. Stopping data collector thread...")
	stop_event.set()
	collector.join()

	logging.info("All processes complete. Exiting script.")

if __name__ == "__main__":
	main() 