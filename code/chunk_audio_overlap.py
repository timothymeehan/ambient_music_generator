# Import necessary libraries
import sys
from pydub import AudioSegment
from pydub import silence

# Function to check for silent chunks
def is_silent(chunk, chunk_length):
    # if at least half of chunk is silence, mark as silent
    silences = silence.detect_silence(chunk,
									  min_silence_len=int(chunk_length/2),
									  silence_thresh=-64)
    if silences:
        return True
    else:
        return False

# Load file and set directories (full_path is argument in cmd line call)
full_path = sys.argv[1]
album_audio = AudioSegment.from_file(full_path)
album_file = full_path.split('/')[-1]
album_name = album_file.replace('.wav', '')
output_path = full_path.replace(album_file, '') + 'chunks/'

# Set interval (chunk size), overlap, get length of album
album_length = len(album_audio)
chunk_length = 8 * 1000
overlap = 4 * 1000 # adjust here depending on desired no. of samples

# Initialize start and end seconds to 0
start = 0
end = 0

# Iterate from 0 to end of the file,
# with increment = chunk_length
cnt = 1
flag = 0 # use to break loop once reach end of album
for i in range(0, 8 * album_length, chunk_length):

	# At first, start is 0, end is the chunk_length
	# Else, start=prvs end-overlap, end=start+chunk_length
	if i == 0:
		start = 0
		end = chunk_length
	else:
		start = end - overlap
		end = start + chunk_length

	# Set flag to 1 if endtime exceeds length of file
	if end >= album_length:
		end = album_length
		flag = 1

	# Storing audio file from the defined start to end
	chunk = album_audio[start:end]
	if flag == 0 and not is_silent(chunk, chunk_length):
		filename = album_name + f'_chunk_{cnt}.wav'
		chunk.export(output_path + filename, format ="wav")
		print("Processing chunk " + str(cnt) + ". Start = "
				+ str(start) + " end = " + str(end))

	cnt = cnt + 1
