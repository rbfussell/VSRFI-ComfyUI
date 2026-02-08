import os
import json
import subprocess

try:
    import server
    import folder_paths
    web = server.web

    @server.PromptServer.instance.routes.get("/vsrfi/video_info")
    async def get_video_info(request):
        filename = request.query.get("filename", "")
        file_type = request.query.get("type", "input")

        if not filename:
            return web.json_response({"error": "no filename"}, status=400)

        file_path = folder_paths.get_annotated_filepath(filename)
        if not os.path.exists(file_path):
            return web.json_response({"error": "file not found"}, status=404)

        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,nb_frames',
                '-show_entries', 'format=duration',
                '-of', 'json',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            data = json.loads(result.stdout)

            stream = data.get('streams', [{}])[0]

            # Parse FPS from r_frame_rate (e.g. "30/1", "24000/1001")
            fps_str = stream.get('r_frame_rate', '30/1')
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 30.0

            # Frame count and duration
            nb_frames = int(stream.get('nb_frames', 0))
            duration = float(data.get('format', {}).get('duration', 0))

            # If nb_frames is 0, estimate from duration
            if nb_frames == 0 and duration > 0 and fps > 0:
                nb_frames = int(duration * fps)

            return web.json_response({
                'fps': round(fps, 3),
                'frame_count': nb_frames,
                'duration': round(duration, 3)
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

except ImportError:
    pass
