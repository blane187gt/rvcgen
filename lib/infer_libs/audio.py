import librosa
import numpy as np
import av
from io import BytesIO
import ffmpeg
import os
import traceback
import sys
import random
import subprocess

platform_stft_mapping = {
    'linux': os.path.join(os.getcwd(), 'stftpitchshift'),
    'darwin': os.path.join(os.getcwd(), 'stftpitchshift'),
    'win32': os.path.join(os.getcwd(), 'stftpitchshift.exe'),
}

stft = platform_stft_mapping.get(sys.platform)

def wav2(i, o, format):
    inp = av.open(i, 'rb')
    if format == "m4a": format = "mp4"
    out = av.open(o, 'wb', format=format)
    if format == "ogg": format = "libvorbis"
    if format == "mp4": format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame): out.mux(p)

    for p in ostream.encode(None): out.mux(p)

    out.close()
    inp.close()

def audio2(i, o, format, sr):
    inp = av.open(i, 'rb')
    out = av.open(o, 'wb', format=format)
    if format == "ogg": format = "libvorbis"
    if format == "f32le": format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame): out.mux(p)

    out.close()
    inp.close()

def load_audion(file, sr):
    try:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                return np.frombuffer(out.getvalue(), np.float32).flatten()

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

def load_audio(file, sr, DoFormant=False, Quefrency=1.0, Timbre=1.0):
    converted = False
    formanted = False
    file = file.strip(' \n"')
    if not os.path.exists(file):
        raise RuntimeError(
            "Wrong audio path, that does not exist."
        )

    try:
        if not file.endswith(".wav"):
            converted = True
            formatted_file = f"{os.path.splitext(os.path.basename(file))[0]}.wav"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-i", file, formatted_file],
                capture_output=True,
                text=True,
            )
            file = formatted_file
            print(f"File formatted to wav format: {file}\n")

        if DoFormant:
            print("Starting formant shift. Please wait as this process takes a while.")
            formanted_file = f"{os.path.splitext(os.path.basename(file))[0]}_formanted{os.path.splitext(os.path.basename(file))[1]}"
            command = (
                f'{stft} -i "{file}" -q "{Quefrency}" '
                f'-t "{Timbre}" -o "{formanted_file}"'
            )
            subprocess.run(command, shell=True)
            file = formanted_file
            print(f"Formanted {file}\n")

        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                audio_data = np.frombuffer(out.getvalue(), np.float32).flatten()

        if converted:
            try:
                os.remove(formatted_file)
            except Exception as error:
                print(f"Couldn't remove converted type of file due to {error}")
                error = None
            converted = False

        return audio_data

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)
    except Exception:
        raise RuntimeError(traceback.format_exc())

def check_audio_duration(file):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        probe = ffmpeg.probe(file)

        duration = float(probe['streams'][0]['duration'])

        if duration < 0.76:
            print(
                f"Audio file, {file.split('/')[-1]}, under ~0.76s detected - file is too short. Target at least 1-2s for best results."
            )
            return False

        return True
    except Exception as e:
        raise RuntimeError(f"Failed to check audio duration: {e}")