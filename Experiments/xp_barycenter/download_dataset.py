import requests, os, bs4
from bs4 import BeautifulSoup

### List all links to NetCDF files at a given url

def list_nc_datasets(index_url):

    # Parse target url
    reqs = requests.get(index_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # Find all link tags in the page and list their target href
    urls = [] 

    for link in soup.find_all('a'):
        urls.append(link.get('href'))

    # Keep only links to NetCDF file
    nc_data_urls = [x for x in urls if x.endswith('.nc')]

    return nc_data_urls

### Download a file to Google Drive

def download_file_gdrive(index_url, file_url, dest_dir):

    # Create folder
    os.makedirs('./data/'+dest_dir, exist_ok=True)

    # Stream GET request
    r = requests.get(index_url+file_url, stream = True)
    blocks = []  

    # Save the image to folder
    with open(os.path.join('./data/'+dest_dir, os.path.basename(file_url)), "wb") as file:

        for block in r.iter_content(chunk_size = None):
            if block:
                blocks.append(block)

        file.write(b''.join(blocks))

        # Display file size
        file.seek(0, os.SEEK_END)
        print ("Download complete: "+file_url+" – Size: "+str(file.tell())+" bytes.")
        file.close()
    
    return

### Download all NetCDF files rom a target url ###

def download_climate_net(index_url, dest_dir):

    nc_data_urls = list_nc_datasets(index_url)

    for i, file_url in enumerate(nc_data_urls):
        print(str(i+1)+"/"+str(len(nc_data_urls)), end=" ")
        download_file_gdrive(index_url, file_url, dest_dir)

    return


if __name__ == "__main__":
    download_climate_net('https://portal.nersc.gov/project/ClimateNet/climatenet_new/test/', 'test')