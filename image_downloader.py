""" By giving a list of image url's, this script download them all"""



import os, sys
import argparse
import urllib2


def parse_file(infile):

	if infile == None:
		raise NameError("No input file recived")

	url_list = infile.readlines()
	url_list = [item.rstrip('\n') for item in url_list]

	return url_list




def make_path(path, dir_name, infile_name):

	if path == None:
		path = os.path.curdir

	if not os.path.isdir(path):
		raise NameError('The path: %s does not exist'%path)

	if dir_name in [None, '']:
		dir_name = os.path.splitext(infile_name)[0]

	save_path = os.path.join(path, dir_name)

	if not os.path.isdir(save_path):
		os.mkdir(save_path)

	return save_path


def download(url_list, save_path):

	for url in url_list:
		try:
			# open the web page picture and read it into a variable
			opener = urllib2.build_opener()
			page = opener.open(url)
			my_picture = page.read()
		 
			# open file for binary write and save picture
			filename = os.path.join(save_path, url.split('/')[-1])
			fout = open(filename, "wb")
			fout.write(my_picture)
			fout.close()

			if verbose:
				print "Downloaded: ", url
		except urllib2.URLError:
			print "failed to get image: ", url

	
def main():

	parser = argparse.ArgumentParser(description='Download images from a url list')
	parser.add_argument('-p', '--path', dest='save_path', help='path to save files')
	parser.add_argument('-n', '--name', dest='dir_name', help='name of folder for files. Default is same as list file')
	parser.add_argument('list_file', nargs='?', type=argparse.FileType('r'),
					default=sys.stdin, help='file with list of url\'s')
	parser.add_argument('-v', '--verbose', dest='verbose', default=False,
						action='store_true')

	args = parser.parse_args()

	global verbose
	verbose = args.verbose

	url_list = parse_file(args.list_file)
	save_path = make_path(args.save_path, args.dir_name, args.list_file.name)
	download(url_list, save_path)


if __name__ == '__main__':
	main()
