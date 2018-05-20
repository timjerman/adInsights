# Ad Insights

assignment.py - main file that contains the solutions to assignments 1 - 4.2 (uses dataLoader)

onlineDetector.py - the online stream detection (assignment 4.3) (uses dataLoader, streamEngagement, cusumChangeDetector)

streamEngagement.py - reads and analyzes a json file line by line computing the ad engagement on the go (uses dataLoader)

cusumChangeDetector.py - cusum change detector algorithm with an offline example (streamEnagement must be first run to generate the required data)

dataLoader.py - loads and preprocesses the json file (called from other scripts)

dataAnalysis.py - a few analysis to get the understanding of the data
