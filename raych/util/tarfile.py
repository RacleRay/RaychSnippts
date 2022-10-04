import os
import shutil
import tarfile
import tempfile
import zipfile


def extract_tar(archive_file, target_dir=None, delete=False):
    """
    解压一个 gz file
    :param archive_file: 文件路径
    :param target_dir: 目标文件夹路径
    :param delete: 是否删除原文件
    """
    # default to same directory as tar file
    if not target_dir:
        target_dir, _ = os.path.split(archive_file)
    try:
        with tarfile.open(archive_file, 'r:gz') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, target_dir)
    finally:
        if delete:
            os.remove(archive_file)


def tar_directory(target_dir: str, archive_filename: str) -> str:
    """
    压缩一个文件夹.
    :param target_dir: 要压缩的文件夹
    :param archive_filename: 目标 tar file 的名称
    :return: 目标 tar file 的路径
    """
    tar_file = os.path.join(target_dir, archive_filename)
    tar_directories(
        target_dirs_to_archive_paths={target_dir: '.'},
        tarfile_path=tar_file,
    )
    return tar_file


def tar_directories(target_dirs_to_archive_paths, tarfile_path):
    """
    压缩文件夹
    """
    with tarfile.open(tarfile_path, 'w:gz') as tar:
        for dir_path, archive_name in target_dirs_to_archive_paths.items():
            target_dir = os.path.normpath(dir_path)
            tar.add(target_dir, arcname=archive_name)


def unzip_directory(archive_file: str,
                    target_dir: str = None,
                    delete: bool = False):
    """
    解压 zip file.
    :param archive_file: 文件路径
    :param target_dir: 目标文件夹路径
    :param delete: 是否删除原文件
    """
    if not target_dir:
        target_dir, _ = os.path.split(
            archive_file)  # default to same directory as archive file
    shutil.unpack_archive(archive_file, target_dir, 'zip')
    if delete:
        os.remove(archive_file)


def zip_directory(target_dir: str, archive_filename: str) -> str:
    """
    Zip 一个文件夹
    :return: zip archive file的路径
    """
    # 在临时位置创建存档，然后将其移动到目标目录
    # 否则，生成的文件将包含一个额外的零字节文件.
    target_path = os.path.join(target_dir, archive_filename)
    with tempfile.TemporaryDirectory() as temp_dirpath:
        tmp_zip_filename = os.path.join(
            temp_dirpath, 'clusterrunner_tmp__' + archive_filename)
        with zipfile.ZipFile(tmp_zip_filename,
                             'w',
                             compression=zipfile.ZIP_DEFLATED) as zf:
            for dirpath, dirnames, filenames in os.walk(target_dir):
                for filename in filenames:
                    path = os.path.normpath(os.path.join(dirpath, filename))
                    if os.path.isfile(path):
                        relpath = os.path.relpath(path, target_dir)
                        zf.write(path, relpath)
        shutil.move(tmp_zip_filename, target_path)
    return target_path
