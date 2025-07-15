import time
import hashlib
import random
import string
    
    
def TimeProvider() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def DateProvider() -> str:
    return time.strftime('%Y-%m-%d', time.localtime())


def RandomHashProvider() -> str:
    random.seed(time.time())
    return hashlib.md5(str(random.random()).encode('utf-8')).hexdigest()