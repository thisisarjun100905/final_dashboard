import pymysql

def create_conn_240():
    return pymysql.connect(
        host='192.168.24.240',
        port=3306,
        user='arjun_10176189',
        passwd='@Rju^_89',
        db='datagrowth',
        charset='utf8',
        local_infile=1 
    )

def create_conn_120():
    return pymysql.connect(
        host='192.168.39.120',
        port=3306,
        user='arjun_10176189',
        passwd='@Rju^_89',
        db='datagrowth',
        charset='utf8',
        local_infile=1 
    )

def create_conn_216():
    return pymysql.connect(
        host='172.29.0.216',
        port=3306,
        user='arjun_10176189',
        passwd='@Rju^_89',
        db='db_paidcontracts',
        charset='utf8',
        local_infile=1 
    )
__all__ = ['create_conn_240', 'create_conn_120', 'create_conn_216']