import { CSSObject } from "@emotion/react";

const styles = <CSSObject>{
  "*": {
    boxSizing: "border-box",
  },
  body: {
    margin: 0,
    boxSizing: "border-box",
    fontFamily: "'Roboto', sans-serif",
  },
  "h1, h2, h3, h4, h5, h6, p": {
    margin: 0,
  },
  a: {
    textDecoration: "none",
    color: "inherit",
    "&:hover": {
      textDecoration: "underline",
    },
  },
};
export default styles;
