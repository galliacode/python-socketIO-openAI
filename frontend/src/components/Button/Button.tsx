import { styled } from "@mui/material/styles";
import Button, { ButtonProps } from "@mui/material/Button";

export default styled(Button)<ButtonProps>(() => ({
  textTransform: "none",
}));
