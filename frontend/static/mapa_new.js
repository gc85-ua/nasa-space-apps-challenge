// Open-world exoplanet explorer: fly freely from Earth to distant planets
import * as THREE from 'three'
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js'

;(function(){
  const container = document.getElementById('canvas-container')
  const tooltip = document.getElementById('tooltip')
  const btnEarth = document.getElementById('btn-earth')

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a15)
  scene.fog = new THREE.Fog(0x0a0a15, 1, 3000)

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000)
  camera.position.set(0, 5, 15)

  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(window.devicePixelRatio || 1)
  container.appendChild(renderer.domElement)

  // PointerLock controls for first-person free flight
  const controls = new PointerLockControls(camera, renderer.domElement)
  
  renderer.domElement.addEventListener('click', () => {
    controls.lock()
  })

  // Movement state
  const moveState = { forward: false, backward: false, left: false, right: false, up: false, down: false }
  const velocity = new THREE.Vector3()
  const direction = new THREE.Vector3()
  const moveSpeed = 50

  // Keyboard controls
  document.addEventListener('keydown', (e) => {
    switch(e.code) {
      case 'KeyW': case 'ArrowUp': moveState.forward = true; break
      case 'KeyS': case 'ArrowDown': moveState.backward = true; break
      case 'KeyA': case 'ArrowLeft': moveState.left = true; break
      case 'KeyD': case 'ArrowRight': moveState.right = true; break
      case 'Space': moveState.up = true; break
      case 'ShiftLeft': case 'ShiftRight': moveState.down = true; break
    }
  })
  document.addEventListener('keyup', (e) => {
    switch(e.code) {
      case 'KeyW': case 'ArrowUp': moveState.forward = false; break
      case 'KeyS': case 'ArrowDown': moveState.backward = false; break
      case 'KeyA': case 'ArrowLeft': moveState.left = false; break
      case 'KeyD': case 'ArrowRight': moveState.right = false; break
      case 'Space': moveState.up = false; break
      case 'ShiftLeft': case 'ShiftRight': moveState.down = false; break
    }
  })

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.6)
  scene.add(ambient)
  const sunLight = new THREE.PointLight(0xffffdd, 2.5, 3000)
  sunLight.position.set(0, 0, 0)
  scene.add(sunLight)

  // EARTH at origin
  const earthGeom = new THREE.SphereGeometry(6, 64, 64)
  const earthMat = new THREE.MeshStandardMaterial({ 
    color: 0x1e90ff,
    metalness: 0.2, 
    roughness: 0.7
  })
  const earth = new THREE.Mesh(earthGeom, earthMat)
  earth.position.set(0, 0, 0)
  earth.userData = { name: '🌍 Tierra', radius: '1.0 R⊕', period: '365d', distance: '0 ly' }
  scene.add(earth)

  // Background starfield
  const starsGeo = new THREE.BufferGeometry()
  const starCount = 50000
  const positions = new Float32Array(starCount * 3)
  for(let i=0; i<starCount; i++){
    const r = 2000 + Math.random() * 7000
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    positions[i*3] = r * Math.sin(phi) * Math.cos(theta)
    positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta)
    positions[i*3+2] = r * Math.cos(phi)
  }
  starsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  
  const starsMat = new THREE.PointsMaterial({ 
    color: 0xffffff,
    size: 0.4, 
    transparent: true, 
    opacity: 0.8,
    sizeAttenuation: false
  })
  const starPoints = new THREE.Points(starsGeo, starsMat)
  scene.add(starPoints)

  // Planet types with colors
  const planetTypes = {
    rocky: { 
      colors: [0xb87333, 0x8b4513, 0xa0522d, 0xcd853f],
      roughness: 0.9, 
      metalness: 0.1 
    },
    oceanic: { 
      colors: [0x1e90ff, 0x4169e1, 0x0066cc, 0x0080ff],
      roughness: 0.3, 
      metalness: 0.4 
    },
    gasGiant: { 
      colors: [0xffa500, 0xffcc66, 0xff9933, 0xdaa520],
      roughness: 0.5, 
      metalness: 0.2 
    },
    iceGiant: { 
      colors: [0x87ceeb, 0xb0e0e6, 0xadd8e6, 0x87cefa],
      roughness: 0.4, 
      metalness: 0.3 
    },
    lava: { 
      colors: [0xff4500, 0xff6347, 0xff0000, 0xdc143c],
      roughness: 0.6, 
      metalness: 0.5, 
      emissive: true
    },
    desert: { 
      colors: [0xedc9af, 0xd2b48c, 0xf4a460, 0xdeb887],
      roughness: 0.8, 
      metalness: 0.1 
    },
    frozen: { 
      colors: [0xf0f8ff, 0xe6e6fa, 0xfffafa, 0xf5f5f5],
      roughness: 0.3, 
      metalness: 0.2 
    }
  }

  // Exoplanet data
  const exoplanetData = [
    { name: 'Tatooine (HD 40307 g)', distLY: 42, size: 8.5, type: 'desert' },
    { name: 'Hoth (Gliese 581 d)', distLY: 20.4, size: 7.9, type: 'frozen' },
    { name: 'Bespin (51 Pegasi b)', distLY: 50.9, size: 12.3, type: 'gasGiant' },
    { name: 'Mustafar (CoRoT-7b)', distLY: 489, size: 6.1, type: 'lava' },
    { name: 'Kamino (TRAPPIST-1e)', distLY: 39.5, size: 4.6, type: 'oceanic' },
    { name: 'Endor (Kepler-186f)', distLY: 580, size: 5.9, type: 'oceanic' },
    { name: 'Proxima Centauri b', distLY: 4.24, size: 5.2, type: 'rocky' },
    { name: 'Barnard\'s Star b', distLY: 5.96, size: 4.8, type: 'frozen' },
    { name: 'TRAPPIST-1f', distLY: 39.5, size: 5.1, type: 'iceGiant' },
    { name: 'TRAPPIST-1g', distLY: 39.5, size: 5.4, type: 'oceanic' },
    { name: 'Kepler-442b', distLY: 1206, size: 6.7, type: 'oceanic' },
    { name: 'Kepler-452b', distLY: 1400, size: 7.2, type: 'rocky' },
    { name: 'GJ 667 Cc', distLY: 23.6, size: 7.7, type: 'rocky' },
    { name: 'Tau Ceti e', distLY: 11.9, size: 9.2, type: 'desert' },
    { name: 'Gliese 832 c', distLY: 16.1, size: 7.1, type: 'rocky' },
    { name: 'Kepler-62f', distLY: 1200, size: 6.4, type: 'oceanic' },
    { name: 'Kepler-22b', distLY: 640, size: 9.8, type: 'gasGiant' },
    { name: 'HD 85512 b', distLY: 36, size: 6.2, type: 'rocky' },
    { name: 'Gliese 163 c', distLY: 49, size: 8.3, type: 'lava' },
    { name: 'Kepler-69c', distLY: 2700, size: 8.1, type: 'gasGiant' },
    { name: 'Wolf 1061c', distLY: 13.8, size: 6.8, type: 'rocky' },
    { name: 'Kepler-1229b', distLY: 870, size: 5.5, type: 'oceanic' },
    { name: 'LHS 1140 b', distLY: 40, size: 7.3, type: 'rocky' },
    { name: 'K2-18b', distLY: 124, size: 9.1, type: 'iceGiant' },
    { name: 'Ross 128 b', distLY: 11, size: 5.7, type: 'rocky' },
    { name: 'Teegarden b', distLY: 12.5, size: 5.2, type: 'rocky' },
    { name: 'YZ Ceti b', distLY: 12.1, size: 4.9, type: 'lava' },
    { name: 'Kapteyn b', distLY: 12.8, size: 7.4, type: 'frozen' },
    { name: 'Kepler-1649c', distLY: 300, size: 5.8, type: 'oceanic' },
    { name: 'HD 209458 b', distLY: 159, size: 13.1, type: 'gasGiant' },
    { name: 'WASP-12b', distLY: 871, size: 14.2, type: 'lava' },
    { name: 'Kepler-16b', distLY: 245, size: 11.5, type: 'gasGiant' },
    { name: 'HD 189733 b', distLY: 64.5, size: 12.8, type: 'gasGiant' },
    { name: 'Kepler-10b', distLY: 608, size: 6.5, type: 'lava' },
    { name: 'Gliese 436 b', distLY: 31.8, size: 10.2, type: 'iceGiant' },
    { name: 'HAT-P-11b', distLY: 123, size: 10.8, type: 'iceGiant' },
    { name: 'Kepler-138d', distLY: 219, size: 5.3, type: 'oceanic' },
    { name: 'TOI-700 d', distLY: 101.4, size: 6.1, type: 'oceanic' },
    { name: 'L 98-59 c', distLY: 35, size: 5.9, type: 'rocky' },
    { name: 'GJ 357 d', distLY: 31, size: 7.6, type: 'frozen' },
    { name: 'Kepler-1652b', distLY: 822, size: 6.9, type: 'desert' },
    { name: 'LP 890-9c', distLY: 105, size: 6.3, type: 'oceanic' },
    { name: 'TOI-1452 b', distLY: 100, size: 7.8, type: 'oceanic' },
    { name: 'Kepler-283c', distLY: 1743, size: 7.2, type: 'rocky' }
  ]

  const planets = []

  exoplanetData.forEach((data) => {
    let sceneDistance = data.distLY * 10
    if (data.distLY > 100) {
      sceneDistance = 1000 + Math.log10(data.distLY) * 150
    }

    const typeData = planetTypes[data.type]
    const colorIndex = Math.floor(Math.random() * typeData.colors.length)
    const planetColor = typeData.colors[colorIndex]

    const geo = new THREE.SphereGeometry(data.size, 48, 48)
    const mat = new THREE.MeshStandardMaterial({ 
      color: planetColor,
      metalness: typeData.metalness, 
      roughness: typeData.roughness
    })
    
    if (typeData.emissive) {
      mat.emissive = new THREE.Color(planetColor)
      mat.emissiveIntensity = 0.5
    }
    
    const mesh = new THREE.Mesh(geo, mat)
    
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    mesh.position.set(
      sceneDistance * Math.sin(phi) * Math.cos(theta),
      (Math.random() - 0.5) * sceneDistance * 0.4,
      sceneDistance * Math.cos(phi)
    )

    mesh.userData = {
      name: data.name,
      radius: (data.size / 6).toFixed(2) + ' R⊕',
      period: Math.round(50 + Math.random() * 800) + 'd',
      distance: data.distLY.toFixed(1) + ' ly',
      type: data.type.charAt(0).toUpperCase() + data.type.slice(1)
    }
    
    scene.add(mesh)
    planets.push(mesh)
  })

  // Raycaster for hover tooltips
  const raycaster = new THREE.Raycaster()
  raycaster.far = 3000
  const screenCenter = new THREE.Vector2(0, 0)
  let hovered = null

  function checkHover(){
    if (!controls.isLocked) return
    
    raycaster.setFromCamera(screenCenter, camera)
    const intersects = raycaster.intersectObjects([...planets, earth])
    
    if (intersects.length > 0) {
      const obj = intersects[0].object
      if (obj !== hovered) {
        hovered = obj
        showTooltip(obj.userData)
      }
    } else {
      if (hovered) {
        hovered = null
        hideTooltip()
      }
    }
  }

  function showTooltip(data){
    const typeText = data.type ? ` &nbsp;|&nbsp; Tipo: ${data.type}` : ''
    tooltip.innerHTML = `<strong>${data.name}</strong><br/>Radio: ${data.radius} &nbsp;|&nbsp; Periodo: ${data.period} &nbsp;|&nbsp; Distancia: ${data.distance}${typeText}`
    tooltip.style.left = '50%'
    tooltip.style.top = '80px'
    tooltip.style.transform = 'translateX(-50%)'
    tooltip.classList.add('show')
  }
  
  function hideTooltip(){
    tooltip.classList.remove('show')
  }

  // Return to Earth button
  btnEarth.addEventListener('click', () => {
    camera.position.set(0, 5, 15)
    camera.lookAt(0, 0, -100)
  })
  btnEarth.addEventListener('mouseenter', () => {
    btnEarth.style.transform = 'scale(1.05)'
    btnEarth.style.background = '#0f0fc8'
  })
  btnEarth.addEventListener('mouseleave', () => {
    btnEarth.style.transform = 'scale(1)'
    btnEarth.style.background = '#1313ec'
  })

  // Planet search
  const searchInput = document.getElementById('search-input')
  const searchResults = document.getElementById('search-results')
  
  function navigateToPlanet(planet) {
    const targetPos = planet.position.clone()
    const offset = new THREE.Vector3(0, 5, planet.geometry.parameters.radius * 3)
    camera.position.copy(targetPos).add(offset)
    camera.lookAt(targetPos)
    
    showTooltip(planet.userData)
    setTimeout(hideTooltip, 5000)
  }
  
  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim()
    searchResults.innerHTML = ''
    
    if (query.length < 2) return
    
    const matches = [...planets, earth].filter(p => 
      p.userData.name.toLowerCase().includes(query)
    ).slice(0, 8)
    
    if (matches.length === 0) {
      searchResults.innerHTML = '<div style="padding:8px; color:#888; font-size:13px;">No se encontraron resultados</div>'
      return
    }
    
    matches.forEach(planet => {
      const item = document.createElement('div')
      item.style.cssText = 'padding:10px; margin:4px 0; background:rgba(255,255,255,0.05); border-radius:6px; cursor:pointer; transition:background 0.2s; font-size:13px;'
      item.innerHTML = `<strong>${planet.userData.name}</strong><br/><small style="color:#aaa;">${planet.userData.distance}</small>`
      
      item.addEventListener('mouseenter', () => {
        item.style.background = 'rgba(19,19,236,0.3)'
      })
      item.addEventListener('mouseleave', () => {
        item.style.background = 'rgba(255,255,255,0.05)'
      })
      item.addEventListener('click', () => {
        navigateToPlanet(planet)
        searchInput.value = ''
        searchResults.innerHTML = ''
      })
      
      searchResults.appendChild(item)
    })
  })

  // Resize
  function onResize(){
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  }
  window.addEventListener('resize', onResize)

  // Animation loop
  const clock = new THREE.Clock()
  function animate(){
    requestAnimationFrame(animate)
    const delta = clock.getDelta()
    
    if (controls.isLocked) {
      velocity.x -= velocity.x * 5.0 * delta
      velocity.z -= velocity.z * 5.0 * delta
      velocity.y -= velocity.y * 5.0 * delta
      
      direction.z = Number(moveState.forward) - Number(moveState.backward)
      direction.x = Number(moveState.right) - Number(moveState.left)
      direction.y = Number(moveState.up) - Number(moveState.down)
      direction.normalize()
      
      if (moveState.forward || moveState.backward) velocity.z -= direction.z * moveSpeed * delta
      if (moveState.left || moveState.right) velocity.x -= direction.x * moveSpeed * delta
      if (moveState.up || moveState.down) velocity.y += direction.y * moveSpeed * delta
      
      controls.moveRight(-velocity.x * delta)
      controls.moveForward(-velocity.z * delta)
      camera.position.y += velocity.y * delta
      
      checkHover()
    }
    
    planets.forEach(p => p.rotation.y += 0.001)
    earth.rotation.y += 0.002
    
    renderer.render(scene, camera)
  }

  onResize()
  animate()

  // Instructions overlay
  const instructions = document.createElement('div')
  instructions.style.position = 'absolute'
  instructions.style.top = '20px'
  instructions.style.left = '50%'
  instructions.style.transform = 'translateX(-50%)'
  instructions.style.padding = '12px 24px'
  instructions.style.background = 'rgba(0,0,0,0.8)'
  instructions.style.color = 'white'
  instructions.style.borderRadius = '8px'
  instructions.style.fontFamily = 'Space Grotesk, sans-serif'
  instructions.style.fontSize = '14px'
  instructions.style.textAlign = 'center'
  instructions.style.zIndex = '100'
  instructions.innerHTML = 'Haz clic para empezar • WASD para moverte • Espacio/Shift para subir/bajar • Ratón para mirar'
  document.body.appendChild(instructions)
  
  controls.addEventListener('lock', () => {
    instructions.style.display = 'none'
  })
  controls.addEventListener('unlock', () => {
    instructions.style.display = 'block'
  })

})()
