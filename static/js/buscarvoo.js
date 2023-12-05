const voltar = document.querySelector('#voltar')

voltar.addEventListener('click', () => {
    document.querySelector('.voo').classList.toggle('show')
    document.querySelector('.busca-form form').classList.toggle('show')

})
