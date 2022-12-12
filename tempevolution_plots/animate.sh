ffmpeg -i temp%03d.png  -c:v libx264 -r 30 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" surface_temp_final.mp4
