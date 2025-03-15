"use client"

import {io} from "socket.io-client";
import {useEffect, useState} from "react";


export default function Home() {

  const socket = io("http://localhost:5000/api/states/listen", {transports: ["websocket"]})
  const [currentState, setCurrentState] = useState<string>("IDLE");

  const stateMap: {[index: string]: {[index: string]: string[]}} = {
    "IDLE": {
      "door_1": [],
      "door_2": [],
      "person": ["hidden"]
    },
    "MOTION_DETECTED": {
      "door_1": [],
      "door_2": [],
      "person": ["verify"]
    },
    "ENTRY GRANTED": {
      "door_1": [],
      "door_2": ["open"],
      "person": ["verify"]
    },
    "ENTRY DETECTED": {
      "door_1": [],
      "door_2": ["close"],
      "person": ["enter"]
    },
    "SECOND VERIFICATION": {
      "door_1": [],
      "door_2": [],
      "person": ["verify-2"]
    },
    "EXIT GRANTED": {
      "door_1": ["open"],
      "door_2": [],
      "person": ["verify-2"]
    },
    "EXIT DETECTED": {
      "door_1": ["close"],
      "door_2": [],
      "person": ["exit"]
    }
  }

  const getStateClass = (entity: string) => {
    return stateMap[currentState]?.[entity]?.join(" ") ?? [];
  }

  useEffect(() => {

    socket.on("state", data => {
      console.log(`Current state: ${data}`)
      setCurrentState(JSON.parse(data).state)
    })

    return () => socket.disconnect()

  }, [])

  return (
    <div className="Dashboard">
      <div className="state">State {currentState}</div>
      <div className="mantrap">
        <div className="doormask-1"/>
        <div className="doormask-2"/>
        <div className={`door-1 ${getStateClass("door_1")}`}/>
        <div className={`door-2 ${getStateClass("door_2")}`}/>
        <div className="sensor-1"/>
        <div className="sensor-2"/>
        <div className={`person ${getStateClass("person")}`}>
          {/*<p>ddd</p>*/}
        </div>
        <div className="wall"/>
      </div>
    </div>
  );
}
