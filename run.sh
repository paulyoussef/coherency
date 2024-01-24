#!/bin/bash
nohup python main.py --model bert-base-uncased--relation P1001 --bs 32 > ./logs/bertbaseuncased_P1001.log &&
nohup python main.py --model bert-base-uncased--relation P101 --bs 32 > ./logs/bertbaseuncased_P101.log &&
nohup python main.py --model bert-base-uncased--relation P103 --bs 32 > ./logs/bertbaseuncased_P103.log &&
nohup python main.py --model bert-base-uncased--relation P106 --bs 32 > ./logs/bertbaseuncased_P106.log &&
nohup python main.py --model bert-base-uncased--relation P108 --bs 32 > ./logs/bertbaseuncased_P108.log &&
nohup python main.py --model bert-base-uncased--relation P127 --bs 32 > ./logs/bertbaseuncased_P127.log &&
nohup python main.py --model bert-base-uncased--relation P1303 --bs 32 > ./logs/bertbaseuncased_P1303.log &&
nohup python main.py --model bert-base-uncased--relation P131 --bs 32 > ./logs/bertbaseuncased_P131.log &&
nohup python main.py --model bert-base-uncased--relation P136 --bs 32 > ./logs/bertbaseuncased_P136.log &&
nohup python main.py --model bert-base-uncased--relation P1376 --bs 32 > ./logs/bertbaseuncased_P1376.log &&
nohup python main.py --model bert-base-uncased--relation P138 --bs 32 > ./logs/bertbaseuncased_P138.log &&
nohup python main.py --model bert-base-uncased--relation P140 --bs 32 > ./logs/bertbaseuncased_P140.log &&
nohup python main.py --model bert-base-uncased--relation P1412 --bs 32 > ./logs/bertbaseuncased_P1412.log
