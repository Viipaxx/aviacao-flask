const buscar = document.querySelector('#buscar')

buscar.addEventListener('click', () => {
    document.querySelector('.voo').classList.remove('show')
    document.querySelector('.busca-form form').classList.add('show')
})

const favoritar = document.querySelector('#favoritar')

favoritar.addEventListener('click', () => {
    document.querySelector('.voo').classList.add('show')
})