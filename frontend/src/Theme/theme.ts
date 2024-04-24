import { createTheme } from "@mui/material";
export const theme = createTheme({
  palette: {
    primary: {
      main: "#507DBC",
    },
    secondary: {
      main: "#F4F4F4",
    },
  },
  components: {
    MuiButtonBase: {
      defaultProps: {},
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true,
        variant: "contained",
      },
    },
    MuiAppBar: {
      defaultProps: {
        elevation: 0,
      },
    },
    MuiTextField: {
      defaultProps: {
        size: "small",
        variant: "outlined",
        InputLabelProps: {
          shrink: true,
        },
      },
    },
  },
});
