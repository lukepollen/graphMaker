docker pull postgres:latest

docker run --name my_postgres -e POSTGRES_PASSWORD=flubunatt -v pgdata:/var/lib/postgresql/data -p 5432:5432 -d postgres