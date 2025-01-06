import argparse
import datetime
import os
import re
import sys
import zipfile

# from urllib import request, error
import requests

from researchpkg.industry_classification.config import (
    DATA_END_QTR,
    DATA_START_QTR,
    SEC_RAW_DATA_DIR,
)

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)

_baseurl = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
_user_agent = "Mozilla"
_download_path = os.path.join(os.getcwd(), "sec_zip")
_text_files = ["sub.txt", "num.txt", "pre.txt", "tag.txt"]
_readme = "readme.htm"
_zip_re = re.compile(r"\d{4}q[1-4]\.zip")

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)


def _request(url):
    """Return HTTPResponse or URLError object after requesting url"""
    try:
        headers = {
            "Accept": "application/x-clarity-gzip",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,de;q=0.6",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }

        return requests.get(url, headers=headers)
    except Exception as e:
        return e


def _test_mkdir(path):
    """Resolve absolute/relative path; mkdirs if path does not exist."""
    if not os.path.isabs(path):
        path = os.path.realpath(os.path.join(os.getcwd(), path))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _qtr_range(startqtr, endqtr):
    """Returns a list of zip file names for calendar qtrs between startqtr & endqtr"""
    yr, endyr = int(startqtr[0:4]), int(endqtr[0:4])
    files = ["%sq%s" % (yr, qtr) for yr in range(yr, endyr + 1) for qtr in range(1, 5)]
    return files[files.index(startqtr) : files.index(endqtr) + 1]


def _download_archive(qtr, path):
    """Download a single zip archive from _baseurl"""
    path = _test_mkdir(path)
    filename = qtr + ".zip"
    url = "%s/%s" % (_baseurl, filename)
    print(url)

    # return 'complete' to indicate successful download, else return URLError object
    with _request(url) as response:
        if response.status_code == 200:
            with open(os.path.join(path, filename), "wb") as file:
                file.write(response._content)
                return "complete"
        else:
            return response


def _extract_archive(path, destination, singledir, rmzip):
    """Extract single archive from _baseurl"""
    zip_filename = os.path.split(path)[1]
    qtr = zip_filename.replace(".zip", "")
    # destination = _test_mkdir(destination)
    destination = os.path.join(destination, qtr) if not singledir else destination

    with zipfile.ZipFile(path) as zip_file:
        # extract .txt files to destination, appending filenames with qtr string
        if singledir:
            zip_file.extractall(destination, _text_files)
            for filename in _text_files:
                basename = filename.replace(".txt", "")
                newname = "%s_%s.txt" % (basename, qtr)
                os.rename(
                    os.path.join(destination, filename),
                    os.path.join(destination, newname),
                )

                # prepend 'readme.htm' with underscore (sort to top of directory)
                if "_" + _readme not in os.listdir(destination):
                    zip_file.extract(_readme, destination)
                    os.rename(
                        os.path.join(destination, _readme),
                        os.path.join(destination, "_" + _readme),
                    )

        # extract to subdirectories mirroring zip archive structure
        else:
            zip_file.extractall(destination)

    if rmzip:
        os.remove(path)


def latest_qtr():
    """Get the latest available quarter from SEC.gov"""
    today = datetime.date.today()
    yr, qtr = today.year, (today.month - 1) // 3
    yr, qtr = (yr - 1, 4) if qtr == 0 else (yr, qtr)
    qtr = "%sq%s" % (yr, qtr)

    # request prior calendar qtr zip from SEC.gov
    with _request("%s/%s.zip" % (_baseurl, qtr)) as response:
        if response.status_code == 200:
            return qtr
        else:
            yr, qtr = (yr - 1, 4) if qtr == 1 else (yr, qtr - 1)
            return "%sq%s" % (yr, qtr)


def download(qtr=None, startqtr="2009q1", endqtr=None, path=None):
    """Download zip archive(s) from SEC.gov"""
    qtr = latest_qtr() if qtr == "latest" else qtr
    path = _download_path if path is None else _test_mkdir(path)
    sys.stdout.write("\nDownloading %s/" % _baseurl)
    sys.stdout.flush()

    # single quarter
    if qtr is not None:
        sys.stdout.write("%s.zip... " % qtr)
        sys.stdout.flush()
        status = _download_archive(qtr, path)
        print(status, flush=True)

    # range of quarters
    else:
        print("...\n")
        endqtr = latest_qtr() if endqtr is None else endqtr
        for qtr in _qtr_range(startqtr, endqtr):
            sys.stdout.write("  %s.zip... " % qtr)
            sys.stdout.flush()
            status = _download_archive(qtr, path)
            print(status, flush=True)


def extract(path=None, destination=None, singledir=False, rmzip=False):
    """Extract sec zip archive(s) located at/within path"""
    path = _download_path if path is None else _test_mkdir(path)
    destination = os.getcwd() if destination is None else _test_mkdir(destination)
    zip_re = re.compile(r"\d{4}q[1-4]\.zip")
    filename = os.path.split(path)[1]

    # path is zip archive
    if zip_re.match(filename):
        sys.stdout.write("\nExtracting %s... " % path)
        sys.stdout.flush()
        _extract_archive(path, destination, singledir, rmzip)
        print("complete")

    # directory path
    else:
        print("\nExtracting %s...\n" % path, flush=True)
        for zip in sorted(os.listdir(path)):
            if zip_re.match(zip):
                sys.stdout.write("  %s... " % zip)
                sys.stdout.flush()
                zip_path = os.path.join(path, zip)
                _extract_archive(zip_path, destination, singledir, rmzip)
                print("complete", flush=True)

        # remove path if empty
        if rmzip:
            try:
                os.rmdir(path)
            except:
                pass


if __name__ == "__main__":
    download(
        startqtr=DATA_START_QTR,
        endqtr=DATA_END_QTR,
        path=SEC_RAW_DATA_DIR,
    )

    extract(
        path=SEC_RAW_DATA_DIR,
        destination=SEC_RAW_DATA_DIR,
        singledir=False,
        rmzip=True,
    )
