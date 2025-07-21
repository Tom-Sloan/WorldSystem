FROM node:22-alpine

WORKDIR /app

# Use environment variables passed at runtime
RUN echo "VITE_API_URL=${VITE_API_URL}" > .env
RUN echo "VITE_WS_HOST=${VITE_WS_HOST}" >> .env

# Install dependencies first
COPY package*.json ./
RUN npm install


EXPOSE 5173
CMD ["npm", "run", "dev"] 