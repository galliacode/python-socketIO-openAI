export default interface CredentialInterface {
  dialect: string;
  host: string;
  port: number;
  username: string;
  password?: string;
  db_name: string;
  driver: string;
  conn_name?: string;
  conn_uuid?: string;
  conn_type_id?: number;
  schema_name?: string;
}
