import os
import csv

path_to_images = './landscapes/'
results_csv = 'full.csv'
keyword_options = ['city', 'field', 'forest', 'mountain', 'ocean', 'lake', 'road']
output_csv = 'cleaned.csv'

def parse_filename_from_url(url):
    clean_name = url.split('.com/')[1].split('/')[1]
    return clean_name

def process_results(csv_file):
    all_images = []
    with open(csv_file, 'rt') as f:
        reader = csv.reader(f)
        results = list(reader)
        header = results[0]
        results.pop(0)
    for i, item in enumerate(header):
        if header[i] == 'emotion_types':
            emotion_types_index = i 
        if header[i] == 'image_url':
            image_url_index = i
        if header[i] == '_worker_id':
            worker_id_index = i
        if header[i] == 'extra_info':
            extra_info_index = i
        if header[i] == 'category':
            category_index = i
        
    with open(output_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for result in results:
            filename = parse_filename_from_url(result[image_url_index])
            file_path = path_to_images + result[category_index] + '/' + filename
            labels = ' '.join(result[emotion_types_index].split('\n'))
            writer.writerow([file_path, labels])
    csvfile.close()

process_results(results_csv)








