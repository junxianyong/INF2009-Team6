"use client"

import {io} from "socket.io-client";
import {useEffect, useState} from "react";


export default function Home() {

    const socket = io(`${process.env.NEXT_PUBLIC_API_URL}/states/listen`, {transports: ["websocket"]});
    const [currentState, setCurrentState] = useState<string>("IDLE");

    const stateMap: { [index: string]: { [index: string]: string[] } } = {
        "IDLE": {
            "door_1": [],
            "door_2": [],
            "person": ["hidden"]
        },
        "VERIFYING_FACE": {
            "door_1": [],
            "door_2": [],
            "person": ["verify"]
        },
        "WAITING_FOR_PASSAGE_G1": {
            "door_1": [],
            "door_2": ["open"],
            "person": ["verify"]
        },
        "CHECKING_MANTRAP": {
            "door_1": [],
            "door_2": ["close"],
            "person": ["enter"]
        },
        "VERIFYING_FACE_G2": {
            "door_1": [],
            "door_2": [],
            "person": ["verify-2"]
        },
        "WAITING_FOR_PASSAGE_G2": {
            "door_1": ["open"],
            "door_2": [],
            "person": ["verify-2"]
        },
        "EXITING": { // THIS STATE IS TO MAKE THE
            "door_1": ["close"],
            "door_2": [],
            "person": ["exit"]
        }
    }

    const getStateClass = (entity: string) => {
        return stateMap[currentState]?.[entity]?.join(" ") ?? [];
    }

    useEffect(() => {
        socket.on("state", (data) => {
            console.log("New state received:", data.state);
            // If the state is not in stateMap ignore the update
            if (!stateMap[data.state]) return;

            // Use the functional form to always have the latest state
            setCurrentState(prevState => {
                if (prevState === "WAITING_FOR_PASSAGE_G2" && data.state === "IDLE") {
                    console.log("Transitioning from WAITING_FOR_PASSAGE_G2 to IDLE");
                    // First show the exit animation
                    setTimeout(() => {
                        console.log("Transitioning to IDLE");
                        setCurrentState("IDLE");
                    }, 3000); // Adjust timing to match your CSS transitions
                    return "EXITING";
                }
                return data.state;
            });
        });

        return () => {
            socket.disconnect();
        };
    }, []);

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
