const validCodes = {
  "0001": ["tf.gather.md", "aug_tf.gather.md"],
  "0002": ["tf.clip_by_global_norm.md", "aug_tf.clip_by_global_norm.md"],
  "0003": ["tf.keras.layers.Dense.md", "aug_tf.keras.layers.Dense.md"],
  "0004": ["tf.data.Dataset.md", "aug_tf.data.Dataset.md"],
  "0005": ["tf.nn.embedding_lookup_sparse.md", "aug_tf.nn.embedding_lookup_sparse.md"],
  "0006": ["tf.keras.layers.RNN.md", "aug_tf.keras.layers.RNN.md"],
  "0007": ["tf.custom_gradient.md", "aug_tf.custom_gradient.md"],
  "0008": ["tf.cond.md", "aug_tf.cond.md"],
  "0009": ["tf.while_loop.md", "aug_tf.while_loop.md"],
  "0010": ["tf.nn.conv1d.md", "aug_tf.nn.conv1d.md"],
  "0011": ["tf.constant.md", "aug_tf.constant.md"],
  "0012": ["tf.map_fn.md", "aug_tf.map_fn.md"],
  "0013": ["tf.keras.metrics.SensitivityAtSpecificity.md", "aug_tf.keras.metrics.SensitivityAtSpecificity.md"],
  "0014": ["tf.nn.sparse_softmax_cross_entropy_with_logits.md", "aug_tf.nn.sparse_softmax_cross_entropy_with_logits.md"],
  "0015": ["tf.feature_column.categorical_column_with_hash_bucket.md", "aug_tf.feature_column.categorical_column_with_hash_bucket.md"],
  "0016": ["tf.feature_column.categorical_column_with_vocabulary_list.md", "aug_tf.feature_column.categorical_column_with_vocabulary_list.md"],
  "0017": ["tf.scatter_nd.md", "aug_tf.scatter_nd.md"],
  "0018": ["tf.data.experimental.make_csv_dataset.md", "aug_tf.data.experimental.make_csv_dataset.md"],
  "0019": ["tf.compat.v1.layers.batch_normalization.md", "aug_tf.compat.v1.layers.batch_normalization.md"],
  "0020": ["tf.pad.md", "aug_tf.pad.md"],
  "0021": ["tf.nn.sampled_softmax_loss.md", "aug_tf.nn.sampled_softmax_loss.md"],
  "0022": ["tf.keras.Model.md", "aug_tf.keras.Model.md"],
  "0023": ["tf.nn.conv2d.md", "aug_tf.nn.conv2d.md"],
  "0024": ["tf.compat.v1.estimator.DNNClassifier.md", "aug_tf.compat.v1.estimator.DNNClassifier.md"],
};



const inputs = document.querySelectorAll(".code-input");
const button = document.querySelector("#proceed-btn");

const urlParams = new URLSearchParams(window.location.search);
const errorMessage = urlParams.get('error');

if (errorMessage) {
  displayErrorMessage("The code you entered is incorrect. Please try again.")
}


// Function to display error message
function displayErrorMessage(message) {
  // Create or select an error message element
  let errorMessage = document.querySelector('.error-message');
  if (!errorMessage) {
      errorMessage = document.createElement('div');
      errorMessage.classList.add('error-message');
      document.body.appendChild(errorMessage);
  }
  errorMessage.textContent = message;

  // Add styling to the error message
  errorMessage.style.display = "block";

  // Hide the error message after 3 seconds
  setTimeout(() => {
      errorMessage.style.display = "none";
  }, 3000);

}

// Function to restrict input to numeric values
function restrictToNumericInput(event) {
  const input = event.target;
  const newValue = input.value.replace(/[^0-9]/g, ''); // Remove non-numeric characters
  if (input.value !== newValue) {
    input.value = newValue;
  }
}


// Function to handle code validation
function handleCodeValidation() {
  const code = Array.from(inputs).map((input) => input.value).join("");

  if (validCodes.hasOwnProperty(code)) {
      // If the code is valid, proceed to the main page
      goToSecondPage(code);
  } else {
      // If the code is invalid, show an error message
      displayErrorMessage("The code you entered is incorrect. Please try again.");
  }
}

// iterate over all inputs
inputs.forEach((input, index1) => {

    input.addEventListener("input", restrictToNumericInput);

    input.addEventListener("keyup", (e) => {
        // This code gets the current input element and stores it in the currentInput variable
        // This code gets the next sibling element of the current input element and stores it in the nextInput variable
        // This code gets the previous sibling element of the current input element and stores it in the prevInput variable
        const currentInput = input,
          nextInput = input.nextElementSibling,
          prevInput = input.previousElementSibling;
        // if the value has more than one character then clear it
        if (currentInput.value.length > 1) {
          currentInput.value = "";
          return;
        }
        // if the next input is disabled and the current value is not empty
        //  enable the next input and focus on it
        if (nextInput && nextInput.hasAttribute("disabled") && currentInput.value !== "") {
          nextInput.removeAttribute("disabled");
          nextInput.focus();
        }
        // if the backspace key is pressed
        if (e.key === "Backspace") {
          // iterate over all inputs again
          inputs.forEach((input, index2) => {
            // if the index1 of the current input is less than or equal to the index2 of the input in the outer loop
            // and the previous element exists, set the disabled attribute on the input and focus on the previous element
            if (index1 <= index2 && prevInput) {
              input.setAttribute("disabled", true);
              input.value = "";
              prevInput.focus();
            }
          });
        }

        if (e.key === "Enter" && !inputs[3].disabled && inputs[3].value !== "") {
            handleCodeValidation();
        }

        //if the fourth input( which index number is 3) is not empty and has not disable attribute then
        //add active class if not then remove the active class.
        if (!inputs[3].disabled && inputs[3].value !== "") {
          button.classList.add("active");
          return;
        }
        button.classList.remove("active");
      });
});

//focus the first input which index is 0 on window load
window.addEventListener("load", () => inputs[0].focus());

// Add click event listener to the button
button.addEventListener("click", handleCodeValidation);

// Define the goToSecondPage function
function goToSecondPage(code) {
    // Redirect to the second page with the code as a URL parameter
    localStorage.setItem('code', code)
    window.location.href = "page.html?code=" + encodeURIComponent(code);
}