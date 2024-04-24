import { Link, useNavigate } from "react-router-dom";
import Button from "../../components/Button/Button";
import TextField from "../../components/TextField/TextField";
import {
  StyledForgotPasswordCard,
  StyledForgotPasswordContainer,
  StyledForgotPasswordCardTitle,
  StyledInputContainer,
  StyledLinkHelper,
} from "./style";
import { useForm } from "react-hook-form";
import { useAuth } from "../../hooks/useAuth";
import { toast } from "react-toastify";

const ForgotPasswordPage = () => {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();
  const { sendPasswordReset } = useAuth();
  const navigate = useNavigate();
  const submitForm = (data: any) => {
    sendPasswordReset(data.email, (response) => {
      if (response.ok) {
        toast.success(
          "Reset email sent! Please check your inbox to proceed further.",
        );
        navigate("/login");
      } else {
        toast.error(response.message);
      }
    });
  };
  return (
    <StyledForgotPasswordContainer>
      <StyledForgotPasswordCard>
        <form onSubmit={handleSubmit(submitForm)}>
          <StyledForgotPasswordCardTitle>
            Forgot password
          </StyledForgotPasswordCardTitle>
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Email"
              placeholder="Enter your email"
              helperText={
                errors.email ? (errors.email.message as string) : null
              }
              error={errors.email ? true : false}
              {...register("email", {
                required: {
                  message: "email is required",
                  value: true,
                },
              })}
            />
          </StyledInputContainer>
          <Button type="submit" fullWidth>
            Send reset email
          </Button>
          <StyledLinkHelper>
            <Link to={"/login"}>Login here!</Link>
          </StyledLinkHelper>
        </form>
      </StyledForgotPasswordCard>
    </StyledForgotPasswordContainer>
  );
};

export default ForgotPasswordPage;
