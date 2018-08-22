import os
import csv

path_to_images = './landscapes/'
keyword_options = ['city', 'field', 'forest', 'mountain', 'ocean', 'lake', 'road']
results_csv = 'full.csv'
output_csv = 'cleaned.csv'

def parse_filename_from_url(url):
    clean_name = url.split('.com/')[1].split('/')[1]
    return clean_name

def process_results(csv_file):   
    with open(csv_file, 'rt') as f:
        reader = csv.reader(f)
        results = list(reader)
        # save the header
        header = results[0]
        # remove the header from the results
        results.pop(0)

    # not using all of these, but helps to identify possible info from the input csv
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

    all_images = {}
    for result in results:
        filename = parse_filename_from_url(result[image_url_index])
        file_path = path_to_images + result[category_index] + '/' + filename
        # emotion tags, separated by a space
        labels = ' '.join(result[emotion_types_index].split('\n'))
        if file_path in all_images.keys():
            all_images[file_path].append(labels)
        else:
            all_images[file_path] = [labels]

    with open(output_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)           
        for key, value in all_images.items():
            emos = []
            for v in value:
                for va in v.split('\r'):
                    emos.append(va.strip())
            writer.writerow([key, ' '.join(emos)])
    csvfile.close()
    print len(results)/3

process_results(results_csv)