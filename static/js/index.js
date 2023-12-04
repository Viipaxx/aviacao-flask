const buttonEntrar = document.querySelector('#entrar')
const usuario = document.querySelector('#usuario')
const senha = document.querySelector('#senha')

buttonEntrar.addEventListener('click', () => {

    if (!senha.value & !usuario.value) {
        alert('Insira os dados...')
        return
    }

    if (!senha.value) {
        alert('Insira a senha...')
        return
    }

    if (!usuario.value) {
        alert('Insira o usuário...')
        return
    }

    buttonEntrar.querySelector('a').href = "/home"

})
