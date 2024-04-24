import React from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { StyledCodeCopyContainer, StyledCopyButton } from "./style";
import { IconButton } from "@mui/material";
import { CopyAllOutlined } from "@mui/icons-material";

interface CodeCopyProps {
  code: string;
  language: string;
}

const CodeCopy: React.FC<CodeCopyProps> = (props) => {
  const code = props.code;
  const language = props.language;
  return (
    <StyledCodeCopyContainer>
      <CopyToClipboard text={code}>
        <StyledCopyButton>
          <IconButton
            sx={{
              borderRadius: "5px",
              backgroundColor: "#ccc",
              "&:hover": {
                backgroundColor: "#ccc",
              },
            }}
          >
            <CopyAllOutlined />
          </IconButton>
        </StyledCopyButton>
      </CopyToClipboard>
      <SyntaxHighlighter language={language} style={coy}>
        {code}
      </SyntaxHighlighter>
    </StyledCodeCopyContainer>
  );
};

export default CodeCopy;
