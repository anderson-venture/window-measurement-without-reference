
WINDOW MEASUREMENT MVP (TRIANGULATION)

STEPS:
1. Add two photos of the SAME window into data/images
2. Measure distance moved between photos (meters)
3. Put that distance into movement.json (baseline_meters)
4. Replace API_KEY in main.py
5. pip install opencv-python inference-sdk
6. python main.py
7. Result saved in data/results

NOTES:
- Works without reference object
- Uses triangulation + known camera focal length
- Accuracy improves with larger movement baseline
