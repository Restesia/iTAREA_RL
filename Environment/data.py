Tasks = {
    0: {
        "taskname":"0",
        "Task_CPUt":150000,
        "Task_RAM":150,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":[""],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    1: {
        "taskname":"1",
        "Task_CPUt":300000,
        "Task_RAM":200,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":[""],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    2: {
        "taskname":"2",
        "Task_CPUt":100000,
        "Task_RAM":75,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":["camera"],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    3: {
        "taskname":"3",
        "Task_CPUt":1000,
        "Task_RAM":75,
        "user":0,
        "MinimTrans":0,
        "sensreq":["thermometer"],
        "periphreq":[""],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    4: {
        "taskname":"name",
        "Task_CPUt":150000,
        "Task_RAM":150,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":[""],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    5: {
        "taskname":"name",
        "Task_CPUt":3000000,
        "Task_RAM":2000,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":["gpu"],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    },
    6: {
        "taskname":"name",
        "Task_CPUt":150000,
        "Task_RAM":150,
        "user":0,
        "MinimTrans":0,
        "sensreq":[""],
        "periphreq":[""],
        "transmit":[""],
        "exlocation":"none",
        "tasktype":"computing",
        "DReq":100
    }
}

Nodes = {
    0: {
        "name":"0",
        "cpu":10000000,
        "bwup":1500000000,
        "pwup":0.3,
        "maxenergy":90,
        "ram":4000,
        "importance":1,
        "pwdown":0.7,
        "bwdown":150000000,
        "sensingunits":[""],
        "peripherials":[""],
        "typecore":"computing",
        "location":"dispatch 2.30",
        "cores":2,
        "percnormal":30
    },
    1: {
        "name":"1",
        "cpu":10000000,
        "bwup":2300000000,
        "pwup":0.3,
        "maxenergy":750,
        "ram":4000,
        "importance":1,
        "pwdown":0.7,
        "bwdown":150000000,
        "sensingunits":[""],
        "peripherials":["gpu"],
        "typecore":"computing",
        "location":"dispatch 2.30",
        "cores":4,
        "percnormal":30
    },
    2: {
        "name":"2",
        "cpu":10000000,
        "bwup":1400000000,
        "pwup":0.3,
        "maxenergy":75,
        "ram":2000,
        "importance":0,
        "pwdown":0.7,
        "bwdown":150000000,
        "sensingunits":["thermometer"],
        "peripherials":[""],
        "typecore":"computing",
        "location":"dispatch 2.30",
        "cores":2,
        "percnormal":30
    },
    3: {
        "name":"3",
        "cpu":10000000,
        "bwup":3000000000,
        "pwup":0.3,
        "maxenergy":450,
        "ram":12000,
        "importance":1,
        "pwdown":0.7,
        "bwdown":150000000,
        "sensingunits":[""],
        "peripherials":["camera"],
        "typecore":"computing",
        "location":"dispatch 2.30",
        "cores":8,
        "percnormal":30
    }
}