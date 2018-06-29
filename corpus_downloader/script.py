import argparse
import logging
import sys

from corpus_downloader import settings
from corpus_downloader.image_corpus.ImageCorpus import ImageCorpus
from .auth import Auth


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.staging:
        settings.GlobalSetting.IS_STAGING = True
    auth = Auth(user=args.user, password=args.password)

    if args.command == 'download':
        assert args.tile_y * args.tile_x < 100, "can't handle this much tile"
        ImageCorpus.from_id(corpus_id=args.corpus_id, auth=auth, output=args.output).download(test_size=args.test_size, width=args.width,
                                                                                            height=args.height, drop_smaller=args.drop_smaller,
                                                                                            tile_x=args.tile_x, tile_y=args.tile_y)
    elif args.command == 'upload':
        ImageCorpus.from_name(corpus_name=args.corpus_name, label_type=args.label_type, auth=auth).upload(corpus_path=args.corpus_path, confidentiality=args.confidentiality, width=args.width,
                                                                                            height=args.height, drop_smaller=args.drop_smaller)

def parse_args():
    auth = argparse.ArgumentParser(add_help=False)
    auth_grp = auth.add_argument_group(title='authentication')
    auth_grp.add_argument("--user", "-u", required=True, type=str, help='user for auth')
    auth_grp.add_argument("--password", "-p", required=True, type=str, help='password for auth')

    generic = argparse.ArgumentParser(add_help=False)
    generic_grp = generic.add_argument_group('generic')
    generic_grp.add_argument("--verbose", "-v", help="Debug mode", action="store_true")
    generic_grp.add_argument("--staging", action='store_true', help="use the staging env")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="available commands", dest='command')

    downloader = subparsers.add_parser('download', help="Download a corpus to disk", parents=[auth, generic])
    downloader.add_argument("corpus_id", type=str, help="the corpus id")
    downloader.add_argument("--output", "-o", default=None, type=str, help='where to store the dataset. Default to the corpus name.')
    downloader.add_argument("--test-size", type=float, default=0.15, help="fraction of data in the test set. Default to 0.15")
    downloader.add_argument("--width", type=int, default=500, help="Minimum width of image. default to 500")
    downloader.add_argument("--height", type=int, default=500, help="Minimum height of image. default  to 500")
    downloader.add_argument("--drop-smaller", action='store_true', help="drop image smaller than size")
    downloader.add_argument("--tile_x", type=int, default=1, help="Number of tile on the x axis. default to 1")
    downloader.add_argument("--tile_y", type=int, default=1, help="Number of tile on the y axis. default to 1")

    uploader = subparsers.add_parser('upload', help="Upload a corpus to Max-ICS", parents=[auth, generic])
    uploader.add_argument("corpus_name", type=str, help="set the corpus name")
    uploader.add_argument("corpus_path", type=str, help="set the corpus absolute path. Images must be in an Images folder, labels in a Labels folder.")
    uploader.add_argument("confidentiality", type=str, help="set the corpus confidentiality")
    uploader.add_argument("label_type", type=str, help="set the corpus label type. Supported: SEMANTIC_SEGMENTATION")
    uploader.add_argument("--width", type=int, default=500, help="Minimum width of image. default to 500")
    uploader.add_argument("--height", type=int, default=500, help="Minimum height of image. default  to 500")
    uploader.add_argument("--drop-smaller", action='store_true', help="drop image smaller than size")


    args = parser.parse_args()
    if args.command is None:
        parser.print_usage()
        sys.exit(0)
    return args
