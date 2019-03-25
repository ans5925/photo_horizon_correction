import urllib.request

txt_file = open("./train_data/file_urls.txt", "r")
urls = []
idx = 0

while True:
    line = txt_file.readline()
    if not line:
        break
    urls.append(line[:61])
    idx += 1
    urllib.request.urlretrieve(line[:61], './train_data/' + str(idx) + '.jpg')

txt_file.close()