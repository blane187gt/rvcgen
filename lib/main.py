import os
import re
import random
from scipy.io.wavfile import write, read
import numpy as np
import yt_dlp
import argparse
from lib.infer import infer_audio
from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'ytdl/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + '.wav'
        sample_rate, audio_data = read(file_path)
        audio_array = np.asarray(audio_data, dtype=np.int16)

        return sample_rate, audio_array


def roformer_separator(roformer_audio, roformer_model, roformer_output_format, roformer_overlap, roformer_segment_size):
    files_list = []
    directory = "./outputs"
    random_id = str(random.randint(10000, 99999))
    pattern = f"{random_id}"
    os.makedirs("outputs", exist_ok=True)
    write(f'{random_id}.wav', roformer_audio[0], roformer_audio[1])
    prompt = f"audio-separator {random_id}.wav --model_filename=model_bs_roformer_ep_317_sdr_12.9755.ckpt --output_dir=./outputs --output_format={roformer_output_format} --normalization=0.9 --mdxc_overlap={roformer_overlap} --mdxc_segment_size={roformer_segment_size}"
    os.system(prompt)

    for file in os.listdir(directory):
        if re.search(pattern, file):
            files_list.append(os.path.join(directory, file))

    stem1_file = files_list[0]
    stem2_file = files_list[1]

    return stem1_file, stem2_file


def process_song(url, model_name, pitch_change=0, keep_files=False, main_gain=1.0, backup_gain=1.0,
                 inst_gain=1.0, index_rate=0.75, filter_radius=3, rms_mix_rate=0.25,
                 f0_method="fcpe", crepe_hop_length=120, protect=0.33, pitch_change_all=False,
                 reverb_rm_size=0, reverb_wet=0, reverb_dry=0, reverb_damping=0,
                 output_format="wav"):
    
    # Download the audio from the URL
    sample_rate, audio_array = download_audio(url)
    
    # Perform separation using roformer separator
    stem1_file, stem2_file = roformer_separator(
        (sample_rate, audio_array), 
        roformer_model="default", 
        roformer_output_format="wav", 
        roformer_overlap=4, 
        roformer_segment_size=256
    )
    
    # Perform inference
    inferred_audio = infer_audio(
        model_name,
        stem2_file,
        pitch_change,
        f0_method,
        min_pitch="-12",
        max_pitch="12",
        crepe_hop_length,
        index_rate,
        filter_radius,
        rms_mix_rate,
        protect,
        split_infer=False,
        min_silence=500,
        silence_threshold=-50,
        seek_step=1,
        keep_silence=200,
        formant_shift=False,
        quefrency=0,
        timbre=1,
        f0_autotune=False,
        output_format=output_format
    )
    
    # Process the audio (e.g., applying additional effects or adjustments)
    final_audio = AudioSegment.from_file(inferred_audio)
    final_audio = final_audio + main_gain  # Adjust main gain
    
    if reverb_rm_size > 0:
        # Example of applying reverb (this is just a placeholder, real implementation might differ)
        final_audio = final_audio.overlay(final_audio, position=0, gain_during_overlay=reverb_wet)
        final_audio = final_audio + reverb_dry - reverb_damping
    
    cover_path = os.path.join(output_dir, f"processed_{os.path.basename(inferred_audio)}")
    final_audio.export(cover_path, format=output_format)
    
    if not keep_files:
        os.remove(stem1_file)
        os.remove(stem2_file)
        os.remove(inferred_audio)
    
    print(f'[+] Cover generated at {cover_path}')
    return cover_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC V2 CLI")
    parser.add_argument("-url", "--url", type=str, required=True, help="URL of the audio to process")
    parser.add_argument("-md", "--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("-pitch", "--pitch_change", type=int, default=0, help="Change in pitch in semitones")
    parser.add_argument("-kf", "--keep_files", type=bool, default=False, help="Keep intermediate files")
    parser.add_argument("-mvol", "--main_vol", type=float, default=1.0, help="Main volume gain")
    parser.add_argument("-bvol", "--backup_vol", type=float, default=1.0, help="Backup volume gain")
    parser.add_argument("-ivol", "--inst_vol", type=float, default=1.0, help="Instrument volume gain")
    parser.add_argument("-irate", "--index_rate", type=float, default=0.75, help="Index rate")
    parser.add_argument("-fradius", "--filter_radius", type=int, default=3, help="Filter radius")
    parser.add_argument("-rms_rate", "--rms_mix_rate", type=float, default=0.25, help="RMS mix rate")
    parser.add_argument("-f0algo", "--pitch_detection_algo", type=str, default="fcpe", help="Pitch detection algorithm")
    parser.add_argument("-hop", "--crepe_hop_length", type=int, default=120, help="Crepe hop length")
    parser.add_argument("-protect", "--protect", type=float, default=0.33, help="Protect")
    parser.add_argument("-pchange_all", "--pitch_change_all", type=bool, default=False, help="Apply pitch change to all")
    parser.add_argument("-rmsize", "--reverb_size", type=int, default 0, help="Reverb room size")
    parser.add_argument("-rwet", "--reverb_wetness", type=float, default=0, help="Reverb wetness")
    parser.add_argument("-rdry", "--reverb_dryness", type=float, default=0, help="Reverb dryness")
    parser.add_argument("-rdamp", "--reverb_damping", type=float, default=0, help="Reverb damping")
    parser.add_argument("-ofmt", "--output_format", type=str, default="wav", help="Output format")

    args = parser.parse_args()
    process_song(args.url, args.model_name, args.pitch_change, args.keep_files, args.main_vol, 
                 args.backup_vol, args.inst_vol, args.index_rate, args.filter_radius, args.rms_mix_rate, 
                 args.pitch_detection_algo, args.crepe_hop_length, args.protect, args.pitch_change_all, 
                 args.reverb_size, args.reverb_wetness, args.reverb_dryness, args.reverb_damping, 
                 args.output_format)
