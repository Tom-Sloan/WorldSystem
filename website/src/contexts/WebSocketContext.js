import { createContext } from "react";

export const WebSocketContext = createContext({
	subscribe: () => {},
	unsubscribe: () => {},
	sendMessage: () => {},
});
