from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence, generate_morph_sequence_with_beats
import numpy as np
import subprocess
import argparse
import shutil
import os
import cv2

def doMorphing(img1, img2, duration, frame_rate, output, img_paths=None, beats_file=None, beat_start=None, beat_end=None):
	[size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2, img_paths)	
	print(f'size: {size}')
	print(f'img1: {len(img1)}')
	print(f'img2: {len(img2)}')
	print(f'points1: {len(points1)}')
	print(f'points2: {len(points2)}')
	# if len(points2)==0:
	# 	return None
	# else:
	# 	print(f'images: {img_paths}')
	# 	np.savetxt("177.csv", points2, delimiter=',', fmt='%i' )
	if len(points1)>76:
		points1=points1[:76]
		print(f'too many points for image: {img_paths[0]}')
	if len(points2)>76:
		points2=points2[:76]
		print(f'too many points for image: {img_paths[1]}')
	tri = make_delaunay(size[1], size[0], list3, img1, img2)
	
	if beats_file and os.path.exists(beats_file):
		print(f"Using beat file: {beats_file} for synchronized morphing")
		generate_morph_sequence_with_beats(beats_file, img1, img2, points1, points2, tri, size, output, beat_start, beat_end)
	else:
		generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, output)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", help="The folder with all images")
	parser.add_argument("--duration", type=int, default=5, help="The duration")
	parser.add_argument("--frame", type=int, default=20, help="The frameame Rate")
	parser.add_argument("--output", help="Output Video Path")
	parser.add_argument("--tmpfolder", default="tmp_videos/", help="The temporary folder to store intermediate videos")
	parser.add_argument("--beats", help="Path to beats CSV file for synchronized morphing")
	parser.add_argument("--beat_interval", type=int, default=1, help="Use every nth beat (default: 1)")
	args = parser.parse_args()

	if(args.folder):
		imgFolder = args.folder
		listImg = sorted(os.listdir(imgFolder))
		
		# New sequential beat handling
		if args.beats and os.path.exists(args.beats):
			# Read the beats file
			import csv
			beats = []
			with open(args.beats, 'r') as f:
				reader = csv.DictReader(f)
				for row in reader:
					beats.append({
						'beat_number': int(row['beat_number']),
						'timestamp': float(row['timestamp']),
						'strength': float(row['strength']) if 'strength' in row else 0.5
					})

			# Apply beat interval - only use beats at the specified interval
			filtered_beats = [beat for i, beat in enumerate(beats) if i % args.beat_interval == 0]
			print(f"Using {len(filtered_beats)} beats out of {len(beats)} total beats (interval: {args.beat_interval})")
				
			# Later in the code, use filtered_beats instead of beats:
			beat_segments = min(len(listImg)-1, len(filtered_beats)-1)



			# Ensure temp folder exists
			if not os.path.exists(args.tmpfolder):
				os.makedirs(args.tmpfolder)
				
			# Create individual morphs for each image pair, synchronized with corresponding beats
			outputList = []			
			for i in range(beat_segments):
				img1_path = os.path.join(imgFolder, listImg[i])
				img2_path = os.path.join(imgFolder, listImg[i+1])
				video_path = args.tmpfolder + f"segment_{i}_{args.output}"
				
				print(f"Processing morph from {listImg[i]} to {listImg[i+1]} using beats {i+1} to {i+2}")
				img1 = cv2.imread(img1_path)
				img2 = cv2.imread(img2_path)
				
				# Use beat i and i+1 for this transition
				beat_start = filtered_beats[i]
				beat_end = filtered_beats[i+1]
				
				doMorphing(img1, img2, args.duration, args.frame, video_path, 
						[listImg[i], listImg[i+1]], beats_file=args.beats, 
						beat_start=beat_start, beat_end=beat_end)
				
				outputList.append(f"file '{video_path}'")
			
			# Write the sequence file for ffmpeg concatenation
			with open('imageslist.txt', 'w') as f:
				f.write('\n'.join(outputList))
				
			print(f"Created beat-synchronized morph sequence with {beat_segments} transitions")
			
		else:
			# Original non-beat processing code
			processed_videos = []
			if os.path.exists('imageslist.txt'):
				with open('imageslist.txt', 'r') as f:
					for line in f:
						if line.startswith("file '"):
							video_path = line.strip()[6:-1]  # Extract path between "file '" and "'"
							processed_videos.append(video_path)
			
			outputList = []
			for i in range(0, len(listImg)-1):
				# Check if video already exists
				video_path = args.tmpfolder + str(i) + "_" + args.output
				if video_path in processed_videos and os.path.exists(video_path):
					print(f"Skipping morphing for {listImg[i]} and {listImg[i+1]} - video already exists")
					outputList.append(f"file '{video_path}'")
				else:
					print("on traite le morphing des images "+listImg[i]+" et "+ listImg[i+1])
					img1 = cv2.imread(os.path.join(imgFolder,listImg[i]))
					img2 = cv2.imread(os.path.join(imgFolder, listImg[i+1]))
					doMorphing(img1, img2, args.duration, args.frame, video_path, [listImg[i],listImg[i+1]])
					outputList.append(f"file '{video_path}'")

			# Write the sequence file for ffmpeg concatenation
			with open('imageslist.txt','w') as f:
				f.write('\n'.join(outputList))
				
		# Merge all the temporary videos into one (final step for both beat and non-beat modes)
		os.system('ffmpeg -f concat -safe 0 -i imageslist.txt -c copy ' + args.output)

#python code/__init__.py --folder aligned_images/ --beats code/beats.csv --output morphed_sequence.mp4