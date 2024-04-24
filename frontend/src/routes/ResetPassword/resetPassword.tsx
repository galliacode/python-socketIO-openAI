import { useForm } from "react-hook-form";
import Button from "../../components/Button/Button";
import TextField from "../../components/TextField/TextField";
import {
  StyledInputContainer,
  StyledResetPasswordCard,
  StyledResetPasswordContainer,
  StyledResetPasswordCardTitle,
} from "./style";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { toast } from "react-toastify";

const ResetPasswordPage = () => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  3;
  const token = searchParams.get("token");
  const { resetPassword } = useAuth();
  const onSubmit = (data: any) => {
    resetPassword(
      {
        token: token!,
        password: data.password,
      },
      (response: any) => {
        if (response.ok) {
          navigate("/login");
          toast.success("Password reset successful!");
        } else {
          toast.error(response.message);
        }
      },
    );
  };
  if (!token) {
    navigate("/");
  }
  return (
    <StyledResetPasswordContainer>
      <StyledResetPasswordCard>
        <form onSubmit={handleSubmit(onSubmit)}>
          <StyledResetPasswordCardTitle>
            Reset password
          </StyledResetPasswordCardTitle>
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Password"
              type="password"
              placeholder="Set new password"
              helperText={
                errors.password ? (errors.password.message as string) : null
              }
              error={errors.password ? true : false}
              {...register("password", {
                required: {
                  message: "Password is required",
                  value: true,
                },
                minLength: {
                  message: "Minimum 8 characters",
                  value: 8,
                },
              })}
            />
          </StyledInputContainer>
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Confirm Password"
              type="password"
              placeholder="Confirm new password"
              helperText={
                errors.confirmPassword
                  ? (errors.confirmPassword.message as string)
                  : null
              }
              error={errors.confirmPassword ? true : false}
              {...register("confirmPassword", {
                required: {
                  message: "Password is required",
                  value: true,
                },
                validate: (val) => {
                  if (watch("password") != val) {
                    return "Passwords do not match";
                  }
                },
              })}
            />
          </StyledInputContainer>
          <Button type="submit" fullWidth>
            Reset password
          </Button>
        </form>
      </StyledResetPasswordCard>
    </StyledResetPasswordContainer>
  );
};

export default ResetPasswordPage;
