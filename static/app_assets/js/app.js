let currentStep = 0;

// onload show first step
showStep(currentStep);

// on next button click
document.getElementById('next').addEventListener('click', function () {
    // Validating the form
    if (validateForm(currentStep)) {

        // incrementing the step
        currentStep += 1;

        // showing the incremented step;
        showStep(currentStep);
    }

});

// on prev button click
document.getElementById('previous').addEventListener('click', function () {

    // decrementing the step
    currentStep -= 1;

    // showing decremented step
    showStep(currentStep);
});

function showStep(step) {
    // display button
    // checking whether the current step is valid or not
    if (step === 0) {
        document.getElementById('previous').style.display = 'none'
        document.getElementById('next').style.display = 'inline';
        document.getElementById('submit').style.display = 'none';
    } else if (step === document.getElementsByClassName('step').length - 1) {
        document.getElementById('previous').style.display = 'inline'
        document.getElementById('next').style.display = 'none';
        document.getElementById('submit').style.display = 'inline';
    } else {
        document.getElementById('previous').style.display = 'inline'
        document.getElementById('next').style.display = 'inline';
        document.getElementById('submit').style.display = 'none';
    }

    // displaying question counter
    document.querySelector('#step-counter h1').textContent = step + 1;

    currentStep = step;

    document.querySelectorAll('.step').forEach((item, index) => {
        // display step
        if (index !== step) {
            item.style.display = 'none';
        } else {
            item.style.display = 'block';
        }
    });

}

// for form validation 
function validateForm(step) {
    let subForm = document.querySelectorAll('.step')[step];
    let flag = true;

    // for radio button input (consent form) validation
    if (subForm.querySelector('input[type=radio]') != null) {
        if (subForm.querySelector("input[type=radio]").checked !== true) {
            alert("Please, agree with the terms.");
            flag = false;
        }
    }

    // for all input['number'] and select tags
    subForm.querySelectorAll("input[type=number], select").forEach((item) => {
        // console.log(item);
        if (item.value === '') {
            item.classList.add('is-invalid');
            flag = false;
        } else {
            item.classList.remove('is-invalid');
            item.classList.add('is-valid');
        }
    });

    // for all input['checkbox']
    if (subForm.querySelector('input[type=checkbox]') != null) {
        let check = false;
        subForm.querySelectorAll('input[type=checkbox]').forEach((item, index) => {
            // Can't skip the process -- show alert if user tries to do so
            if (item.checked === true)
                check = true;
        });

        if (!check) {
            alert("Please check atleast one value.");
            flag = false;
        }
    }

    return flag;
}

// Adding event listener for fetching result
document.querySelector('#submit').addEventListener('click', fetchResult);


async function fetchResult(e) {
    e.preventDefault();

    if (currentStep === document.getElementsByClassName('step').length - 1 && validateForm(currentStep)) {

        console.log("Submitted");

        var messgae_print = $('#message_print').val();

        var rizwan = document.getElementById('mydatas');
        // 	console.log(rizwan)
        let fd = new FormData(rizwan);

        let cough_audio = await fetch(document.getElementsByTagName('audio')[0].src).then(
            r => r.blob()
        );

        fd.append("cough_data", cough_audio, "coughFile.wav");

        $.ajax({
            type: "POST",
            url: 'https://reliefme.azurewebsites.net/data',
            data: fd, // Data sent to server, a set of key/value pairs (i.e. form fields and values)
            contentType: false, // The content type used when sending data to the server.
            cache: false, // To unable request pages to be cached
            processData: false,
            success: function (result) {
                swal({
                    title: 'Result',
                    text: result
                });
            }
        });
    }
}