import React from "react";
import { InputLabel, MenuItem, Select, SelectProps } from "@mui/material";
import { StyledFormControl } from "./style";

interface Option {
  value: string;
  label: string;
}

interface CustomSelectProps extends SelectProps {
  options: Option[];
}

const CustomSelect: React.FC<CustomSelectProps> = (props) => {
  return (
    <StyledFormControl
      sx={{
        marginTop: 1,
      }}
      size="small"
      fullWidth
    >
      <InputLabel disabled={props.disabled} id="simple-select-label">
        {props.label}
      </InputLabel>
      <Select
        labelId="simple-select-label"
        id="simple-select"
        value={props.value}
        label={props.label}
        onChange={props.onChange}
        fullWidth
        disabled={props.disabled}
      >
        {props.options.map((option) => (
          <MenuItem value={option.value}>{option.label}</MenuItem>
        ))}
      </Select>
    </StyledFormControl>
  );
};

export default CustomSelect;
