#!/usr/bin/env python3
"""
Add Audio to Video
Merges an audio file with a video file using ffmpeg
"""
import subprocess
import sys
import os
from pathlib import Path


def add_audio_to_video(video_path, audio_path, output_path=None, replace_audio=True):
    """
    Add audio to a video file using ffmpeg
    
    Args:
        video_path: Path to input video file
        audio_path: Path to audio file (mp3, wav, etc.)
        output_path: Path for output video (optional, auto-generated if None)
        replace_audio: If True, replace existing audio. If False, mix with existing audio
    
    Returns:
        Path to output video file
    """
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Generate output path if not provided
    if output_path is None:
        video_stem = Path(video_path).stem
        video_ext = Path(video_path).suffix
        output_path = f"{video_stem}_with_audio{video_ext}"
    
    # Build ffmpeg command
    if replace_audio:
        # Replace existing audio with new audio
        command = [
            'ffmpeg',
            '-i', video_path,      # Input video
            '-i', audio_path,      # Input audio
            '-c:v', 'copy',        # Copy video codec (no re-encoding)
            '-c:a', 'aac',         # Encode audio to AAC
            '-map', '0:v:0',       # Use video from first input
            '-map', '1:a:0',       # Use audio from second input
            '-shortest',           # End when shortest stream ends
            '-y',                  # Overwrite output file
            output_path
        ]
    else:
        # Mix existing audio with new audio
        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first[aout]',
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_path
        ]
    
    print(f"🎬 Adding audio to video...")
    print(f"   Video: {video_path}")
    print(f"   Audio: {audio_path}")
    print(f"   Output: {output_path}")
    print(f"   Mode: {'Replace' if replace_audio else 'Mix'}")
    print()
    
    try:
        # Run ffmpeg
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Success! Video with audio saved to: {output_path}")
            
            # Show file sizes
            original_size = os.path.getsize(video_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   Original: {original_size:.2f} MB")
            print(f"   Output: {output_size:.2f} MB")
            
            return output_path
        else:
            print(f"❌ Error running ffmpeg:")
            print(result.stderr)
            return None
            
    except FileNotFoundError:
        print("❌ Error: ffmpeg not found!")
        print("   Please install ffmpeg:")
        print("   - Ubuntu/Debian: sudo apt install ffmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/download.html")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Command line interface"""
    if len(sys.argv) < 3:
        print("Usage: python add_audio_to_video.py <video_file> <audio_file> [output_file] [--mix]")
        print()
        print("Examples:")
        print("  python add_audio_to_video.py dashboard_recording.mp4 background_music.mp3")
        print("  python add_audio_to_video.py video.mp4 audio.wav output.mp4")
        print("  python add_audio_to_video.py video.mp4 audio.mp3 output.mp4 --mix")
        print()
        print("Options:")
        print("  --mix    Mix new audio with existing audio (default: replace)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
    mix_audio = '--mix' in sys.argv
    
    result = add_audio_to_video(
        video_path,
        audio_path,
        output_path,
        replace_audio=not mix_audio
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
