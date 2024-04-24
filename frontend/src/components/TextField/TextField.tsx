import { TextField, TextFieldProps } from "@mui/material";
import { styled } from "@mui/material/styles";

export default styled(TextField)<TextFieldProps>(() => ({
  "& .MuiInputLabel-root": {
    position: "relative",
    transform: "none",
  },
  "& legend": {
    width: "0.01px",
  },
  "& p": {
    marginLeft: "0px",
  },
}));
