FROM node:18-alpine as build-stage
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
# Use the build argument to set the environment variable
ARG HOST
ARG FASTAPI_PORT
ENV VITE_SERVER_ADRESS=${HOST}:${FASTAPI_PORT}
RUN npm run build

FROM nginx:alpine
WORKDIR /usr/share/nginx/html
RUN rm -rf ./*
COPY --from=build-stage /app/dist .
COPY --from=build-stage /app/src/nginx.conf /etc/nginx/nginx.conf

ENTRYPOINT ["nginx", "-g", "daemon off;"]
