export const postURLEncoded = async (
  url: string,
  data: any,
  withAuth: boolean = true,
) => {
  const formBody = [];
  for (const property in data) {
    const encodedKey = encodeURIComponent(property);
    const encodedValue = encodeURIComponent(data[property]);
    formBody.push(encodedKey + "=" + encodedValue);
  }
  const formURI = formBody.join("&");
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        ...(withAuth && {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        }),
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
      },
      body: formURI,
    });
    if (response.ok) {
      const json = await response.json();
      return {
        ok: true,
        response: json,
      };
    }
    throw await response.json();
  } catch (e: any) {
    return {
      ok: false,
      message: e.detail || "Something went wrong! Please try again later",
    };
  }
};

export const patchURLEncoded = async (
  url: string,
  data: any,
  withAuth: boolean = true,
) => {
  const formBody = [];
  for (const property in data) {
    const encodedKey = encodeURIComponent(property);
    const encodedValue = encodeURIComponent(data[property]);
    formBody.push(encodedKey + "=" + encodedValue);
  }
  const formURI = formBody.join("&");
  try {
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        ...(withAuth && {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        }),
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
      },
      body: formURI,
    });
    if (response.ok) {
      const json = await response.json();
      return {
        ok: true,
        response: json,
      };
    }
    throw await response.json();
  } catch (e: any) {
    return {
      ok: false,
      message: e.detail || "Something went wrong! Please try again later",
    };
  }
};

export const get = async (url: string, withAuth: boolean = true) => {
  try {
    const response = await fetch(url, {
      headers: {
        ...(withAuth && {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        }),
      },
    });
    if (response.ok) {
      const json = await response.json();
      return {
        ok: true,
        response: json,
      };
    }
    throw await response.json();
  } catch (e: any) {
    console.error(e);
    return {
      ok: false,
      message: e.detail || "Something went wrong! Please try again later",
    };
  }
};

export const post = async (
  url: string,
  data?: any,
  withAuth: boolean = true,
) => {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        ...(withAuth && {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        }),
        "Content-Type": "application/json",
      },
      ...(data && { body: JSON.stringify(data) }),
    });
    const json = await response.json();
    if (response.ok) {
      return {
        ok: true,
        response: json,
      };
    }
    throw json.detail || "Something went wrong! Please try again later";
  } catch (e: any) {
    console.error(e);
    return {
      ok: false,
      message: e,
    };
  }
};

export const deleteMethod = async (url: string, withAuth: boolean = true) => {
  try {
    const response = await fetch(url, {
      method: "DELETE",
      headers: {
        ...(withAuth && {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        }),
      },
    });
    if (response.ok) {
      const json = await response.json();
      return {
        ok: true,
        response: json,
      };
    }
    throw await response.json();
  } catch (e: any) {
    console.error(e);
    return {
      ok: false,
      message: e.detail || "Something went wrong! Please try again later",
    };
  }
};
