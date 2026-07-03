from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
import pandas as pd
import time

VM_IP = ''

ssh_username = 'XXX@XXX.XXX.XXX.XX'
ssh_password = 'ssh-XXXX'


db_name = 'RemoteServer'
db_username = 'XXX'
db_password = 'XXXXXX'

##https://sshtunnel.readthedocs.io/en/latest/index.html
def get_server():
    server = SSHTunnelForwarder(
    VM_IP, 
    ssh_username=ssh_username,
    ssh_password=ssh_password,
    remote_bind_address=('127.0.0.1', 5432)
    )
###() VM_IP, 22)
    server.start()
    engine = create_engine(f'postgresql://{db_username}:{db_password}@127.0.0.1:{server.local_bind.port}/{db_name}')
    return server, engine

def save_to_database():
    df = pd.read_csv('output/data.csv')
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    server, engine = get_server()

    df.to_sql(f'match_data_{timestamp}', engine, if_exists='replace', index=False)
    df = pd.read_csv('output/passes.csv')
    df.to_sql(f'passes_data_{timestamp}', engine, if_exists='replace', index=False)
    server.stop()
