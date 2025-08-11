for file in *.HEIC; do 
    magick "$file" -quality 100 "${file%.HEIC}.jpg"
done
