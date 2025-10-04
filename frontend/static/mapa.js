// Open-world exoplanet explorer with procedural planet textures
import * as THREE from 'three'
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js'

;(function(){
  
  // Function to generate realistic procedural planet textures
  function generatePlanetTexture(type, seed) {
    const canvas = document.createElement('canvas')
    canvas.width = 512
    canvas.height = 512
    const ctx = canvas.getContext('2d')
    
    // Noise function for realistic patterns
    function noise(x, y, freq) {
      return Math.sin(x * freq + seed) * Math.cos(y * freq + seed * 0.7)
    }
    
    const imageData = ctx.createImageData(512, 512)
    const data = imageData.data
    
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        const i = (y * 512 + x) * 4
        const u = x / 512
        const v = y / 512
        
        let r, g, b
        
        if (type === 'rocky') {
          // Rocky planets: browns, reds, grays with craters
          const base = noise(u, v, 8) * 0.3 + noise(u, v, 20) * 0.15
          const craters = Math.max(0, noise(u * 3, v * 3, 50) - 0.7) * 2
          r = 140 + base * 80 + craters * 100
          g = 80 + base * 60 + craters * 60
          b = 50 + base * 40 + craters * 40
        } else if (type === 'oceanic') {
          // Ocean worlds: deep blues with cloud-like swirls
          const clouds = noise(u, v, 12) * 0.4 + noise(u * 2, v * 2, 25) * 0.2
          const ocean = noise(u, v, 6) * 0.3
          r = 20 + clouds * 180 + ocean * 20
          g = 80 + clouds * 150 + ocean * 80
          b = 180 + clouds * 60 + ocean * 40
        } else if (type === 'gasGiant') {
          // Gas giants: banded patterns with storms
          const bands = Math.sin(v * 15 + noise(u, v, 8) * 2) * 0.5 + 0.5
          const storm = Math.max(0, noise(u * 4, v * 4, 20) - 0.6) * 3
          r = 200 + bands * 50 + storm * 50
          g = 140 + bands * 80 + storm * 80
          b = 60 + bands * 40 + storm * 100
        } else if (type === 'iceGiant') {
          // Ice giants: pale blues and teals
          const bands = Math.sin(v * 10 + noise(u, v, 5)) * 0.3 + 0.7
          r = 140 + bands * 80
          g = 200 + bands * 40
          b = 230 + bands * 20
        } else if (type === 'lava') {
          // Lava worlds: reds, oranges with glowing cracks
          const cracks = Math.max(0, noise(u * 6, v * 6, 30) - 0.5) * 2
          const lava = noise(u, v, 8) * 0.3 + 0.7
          r = 200 + lava * 55 + cracks * 255
          g = 50 + lava * 100 + cracks * 80
          b = 0 + cracks * 20
        } else if (type === 'desert') {
          // Desert worlds: sandy yellows and browns
          const dunes = Math.sin(u * 20 + noise(v, u, 10) * 3) * 0.3 + 0.7
          r = 220 + dunes * 35
          g = 180 + dunes * 60
          b = 120 + dunes * 40
        } else if (type === 'frozen') {
          // Frozen worlds: whites and light blues with ice patterns
          const ice = noise(u, v, 15) * 0.2 + noise(u * 3, v * 3, 30) * 0.1 + 0.8
          r = 240 + ice * 15
          g = 245 + ice * 10
          b = 255
        }
        
        data[i] = Math.min(255, Math.max(0, r))
        data[i + 1] = Math.min(255, Math.max(0, g))
        data[i + 2] = Math.min(255, Math.max(0, b))
        data[i + 3] = 255
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
    
    const texture = new THREE.CanvasTexture(canvas)
    texture.needsUpdate = true
    return texture
  }

  const container = document.getElementById('canvas-container')
  const tooltip = document.getElementById('tooltip')
  const btnEarth = document.getElementById('btn-earth')

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000) // Pure black like deep space
  scene.fog = new THREE.Fog(0x000000, 1, 5000)

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000)
  // Start AT Earth position, looking outward into space
  camera.position.set(0, 5, 15)

  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(window.devicePixelRatio || 1)
  container.appendChild(renderer.domElement)

  // PointerLock controls for first-person camera (cursor oculto)
  const controls = new PointerLockControls(camera, renderer.domElement)
  
  // Click to enter pointer lock mode
  renderer.domElement.addEventListener('click', () => {
    controls.lock()
  })

  // Movement state
  const moveState = { forward: false, backward: false, left: false, right: false, up: false, down: false }
  const velocity = new THREE.Vector3()
  const direction = new THREE.Vector3()
  let moveSpeed = 50
  let turboMode = false
  const normalSpeed = 50
  const turboSpeed = 250

  // Keyboard controls
  document.addEventListener('keydown', (e) => {
    switch(e.code) {
      case 'KeyW': case 'ArrowUp': moveState.forward = true; break
      case 'KeyS': case 'ArrowDown': moveState.backward = true; break
      case 'KeyA': case 'ArrowLeft': moveState.left = true; break
      case 'KeyD': case 'ArrowRight': moveState.right = true; break
      case 'Space': moveState.up = true; break
      case 'ControlLeft': case 'ControlRight': moveState.down = true; break
      case 'ShiftLeft': case 'ShiftRight': 
        if (!turboMode) {
          turboMode = true
          moveSpeed = turboSpeed
          updateTurboButton()
        }
        break
    }
  })
  document.addEventListener('keyup', (e) => {
    switch(e.code) {
      case 'KeyW': case 'ArrowUp': moveState.forward = false; break
      case 'KeyS': case 'ArrowDown': moveState.backward = false; break
      case 'KeyA': case 'ArrowLeft': moveState.left = false; break
      case 'KeyD': case 'ArrowRight': moveState.right = false; break
      case 'Space': moveState.up = false; break
      case 'ControlLeft': case 'ControlRight': moveState.down = false; break
      case 'ShiftLeft': case 'ShiftRight':
        if (turboMode) {
          turboMode = false
          moveSpeed = normalSpeed
          updateTurboButton()
        }
        break
    }
  })

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.4)
  scene.add(ambient)
  const sunLight = new THREE.PointLight(0xffffdd, 2, 2000)
  sunLight.position.set(0, 0, 0)
  scene.add(sunLight)

  // EARTH at origin - blue/green colors
  const earthGeom = new THREE.SphereGeometry(6, 64, 64)
  const earthMat = new THREE.MeshStandardMaterial({ 
    color: 0x1e90ff, // Blue as fallback color
    metalness: 0.1, 
    roughness: 0.7
  })
  
  // Try to load Earth texture, but don't fail if it doesn't work
  const textureLoader = new THREE.TextureLoader()
  textureLoader.load(
    'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg',
    (texture) => { earthMat.map = texture; earthMat.needsUpdate = true },
    undefined,
    (error) => { console.log('Earth texture not loaded, using color fallback') }
  )
  
  const earth = new THREE.Mesh(earthGeom, earthMat)
  earth.position.set(0, 0, 0)
  earth.userData = { 
    name: 'Tierra', 
    radius: '1.0 R⊕', 
    mass: '1.0 M⊕',
    distance: '0 años luz', 
    type: 'Rocoso',
    habitable: 'Sí'
  }
  scene.add(earth)

  // Realistic NASA Eyes-style starfield with multiple layers
  
  // Layer 1: Dense Milky Way background (100k+ tiny stars)
  const milkyWayGeo = new THREE.BufferGeometry()
  const milkyWayCount = 120000
  const milkyWayPos = new Float32Array(milkyWayCount * 3)
  const milkyWayColors = new Float32Array(milkyWayCount * 3)
  const milkyWaySizes = new Float32Array(milkyWayCount)
  
  for(let i=0; i<milkyWayCount; i++){
    const r = 3000 + Math.random() * 8000
    const theta = Math.random() * Math.PI * 2
    // Concentrate more stars in a band (Milky Way effect)
    const phi = Math.acos((Math.random() * 2 - 1) * 0.4) 
    
    milkyWayPos[i*3] = r * Math.sin(phi) * Math.cos(theta)
    milkyWayPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta)
    milkyWayPos[i*3+2] = r * Math.cos(phi)
    
    // Slight color variation (white, yellow, blue tints)
    const colorType = Math.random()
    if (colorType < 0.7) {
      milkyWayColors[i*3] = 1.0
      milkyWayColors[i*3+1] = 1.0
      milkyWayColors[i*3+2] = 1.0
    } else if (colorType < 0.9) {
      milkyWayColors[i*3] = 1.0
      milkyWayColors[i*3+1] = 0.95
      milkyWayColors[i*3+2] = 0.8
    } else {
      milkyWayColors[i*3] = 0.8
      milkyWayColors[i*3+1] = 0.9
      milkyWayColors[i*3+2] = 1.0
    }
    
    milkyWaySizes[i] = 0.5 + Math.random() * 0.8
  }
  
  milkyWayGeo.setAttribute('position', new THREE.BufferAttribute(milkyWayPos, 3))
  milkyWayGeo.setAttribute('color', new THREE.BufferAttribute(milkyWayColors, 3))
  milkyWayGeo.setAttribute('size', new THREE.BufferAttribute(milkyWaySizes, 1))
  
  const milkyWayMat = new THREE.PointsMaterial({ 
    vertexColors: true,
    size: 0.8,
    transparent: true, 
    opacity: 0.6,
    sizeAttenuation: false,
    blending: THREE.AdditiveBlending
  })
  const milkyWayPoints = new THREE.Points(milkyWayGeo, milkyWayMat)
  scene.add(milkyWayPoints)
  
  // Layer 2: Medium brightness stars (more visible)
  const mediumStarsGeo = new THREE.BufferGeometry()
  const mediumCount = 8000
  const mediumPos = new Float32Array(mediumCount * 3)
  const mediumColors = new Float32Array(mediumCount * 3)
  const mediumSizes = new Float32Array(mediumCount)
  
  for(let i=0; i<mediumCount; i++){
    const r = 1500 + Math.random() * 6000
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    
    mediumPos[i*3] = r * Math.sin(phi) * Math.cos(theta)
    mediumPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta)
    mediumPos[i*3+2] = r * Math.cos(phi)
    
    // Color variation
    const temp = Math.random()
    if (temp < 0.3) {
      // Blue stars (hot)
      mediumColors[i*3] = 0.7 + Math.random() * 0.3
      mediumColors[i*3+1] = 0.8 + Math.random() * 0.2
      mediumColors[i*3+2] = 1.0
    } else if (temp < 0.7) {
      // White stars
      mediumColors[i*3] = 1.0
      mediumColors[i*3+1] = 1.0
      mediumColors[i*3+2] = 1.0
    } else {
      // Yellow/Orange stars (cool)
      mediumColors[i*3] = 1.0
      mediumColors[i*3+1] = 0.9 + Math.random() * 0.1
      mediumColors[i*3+2] = 0.6 + Math.random() * 0.3
    }
    
    mediumSizes[i] = 1.2 + Math.random() * 1.5
  }
  
  mediumStarsGeo.setAttribute('position', new THREE.BufferAttribute(mediumPos, 3))
  mediumStarsGeo.setAttribute('color', new THREE.BufferAttribute(mediumColors, 3))
  mediumStarsGeo.setAttribute('size', new THREE.BufferAttribute(mediumSizes, 1))
  
  const mediumStarsMat = new THREE.PointsMaterial({ 
    vertexColors: true,
    transparent: true, 
    opacity: 0.9,
    sizeAttenuation: false,
    blending: THREE.AdditiveBlending
  })
  const mediumStarPoints = new THREE.Points(mediumStarsGeo, mediumStarsMat)
  scene.add(mediumStarPoints)
  
  // Layer 3: Bright prominent stars (like visible stars in night sky)
  const brightStarsGeo = new THREE.BufferGeometry()
  const brightCount = 800
  const brightPos = new Float32Array(brightCount * 3)
  const brightColors = new Float32Array(brightCount * 3)
  const brightSizes = new Float32Array(brightCount)
  
  for(let i=0; i<brightCount; i++){
    const r = 800 + Math.random() * 4000
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    
    brightPos[i*3] = r * Math.sin(phi) * Math.cos(theta)
    brightPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta)
    brightPos[i*3+2] = r * Math.cos(phi)
    
    // Prominent color types
    const starType = Math.random()
    if (starType < 0.2) {
      // Blue giants
      brightColors[i*3] = 0.6
      brightColors[i*3+1] = 0.7
      brightColors[i*3+2] = 1.0
    } else if (starType < 0.5) {
      // Bright white
      brightColors[i*3] = 1.0
      brightColors[i*3+1] = 1.0
      brightColors[i*3+2] = 1.0
    } else if (starType < 0.8) {
      // Yellow stars
      brightColors[i*3] = 1.0
      brightColors[i*3+1] = 0.95
      brightColors[i*3+2] = 0.7
    } else {
      // Orange/Red giants
      brightColors[i*3] = 1.0
      brightColors[i*3+1] = 0.7
      brightColors[i*3+2] = 0.4
    }
    
    brightSizes[i] = 2.5 + Math.random() * 3.5
  }
  
  brightStarsGeo.setAttribute('position', new THREE.BufferAttribute(brightPos, 3))
  brightStarsGeo.setAttribute('color', new THREE.BufferAttribute(brightColors, 3))
  brightStarsGeo.setAttribute('size', new THREE.BufferAttribute(brightSizes, 1))
  
  const brightStarsMat = new THREE.PointsMaterial({ 
    vertexColors: true,
    transparent: true, 
    opacity: 1.0,
    sizeAttenuation: false,
    blending: THREE.AdditiveBlending
  })
  const brightStarPoints = new THREE.Points(brightStarsGeo, brightStarsMat)
  scene.add(brightStarPoints)

  // Planet types with properties for procedural generation
  const planetTypes = {
    rocky: { 
      roughness: 0.9, 
      metalness: 0.1 
    },
    oceanic: { 
      roughness: 0.3, 
      metalness: 0.4 
    },
    gasGiant: { 
      roughness: 0.5, 
      metalness: 0.2 
    },
    iceGiant: { 
      roughness: 0.4, 
      metalness: 0.3 
    },
    lava: { 
      roughness: 0.6, 
      metalness: 0.5, 
      emissive: true,
      emissiveColor: 0xff4400
    },
    desert: { 
      roughness: 0.8, 
      metalness: 0.1 
    },
    frozen: { 
      roughness: 0.3, 
      metalness: 0.2 
    }
  }

  // Curated selection of interesting exoplanets (reduced to 18 for better visual experience)
  const exoplanetData = [
    // Star Wars inspired (Easter eggs)
    { name: '🏜️ Tatooine', distLY: 150, size: 8.5, type: 'desert' },
    { name: '❄️ Hoth', distLY: 280, size: 7.9, type: 'frozen' },
    { name: '🌀 Bespin', distLY: 420, size: 13.3, type: 'gasGiant' },
    { name: '🌋 Mustafar', distLY: 650, size: 6.1, type: 'lava' },
    { name: '🌊 Kamino', distLY: 190, size: 7.6, type: 'oceanic' },
    
    // Famous real exoplanets
    { name: 'Proxima Centauri b', distLY: 100, size: 5.2, type: 'rocky' },
    { name: 'TRAPPIST-1e', distLY: 350, size: 4.9, type: 'oceanic' },
    { name: 'Kepler-442b', distLY: 820, size: 6.7, type: 'oceanic' },
    { name: 'Kepler-452b', distLY: 1100, size: 7.2, type: 'rocky' },
    { name: '51 Pegasi b', distLY: 480, size: 12.8, type: 'gasGiant' },
    { name: 'HD 189733 b', distLY: 560, size: 11.8, type: 'gasGiant' },
    { name: 'Kepler-16b', distLY: 720, size: 10.5, type: 'gasGiant' },
    { name: 'CoRoT-7b', distLY: 880, size: 6.5, type: 'lava' },
    { name: 'Gliese 581 d', distLY: 220, size: 7.7, type: 'frozen' },
    { name: 'Kepler-186f', distLY: 950, size: 5.8, type: 'oceanic' },
    { name: 'Ross 128 b', distLY: 140, size: 5.7, type: 'rocky' },
    { name: 'Wolf 1061c', distLY: 170, size: 6.8, type: 'rocky' },
    { name: 'K2-18b', distLY: 390, size: 9.1, type: 'iceGiant' }
  ]

  const planets = []

  exoplanetData.forEach((data, index) => {
    // Increased distances: 15-25 units per light year with stronger logarithmic scaling
    let sceneDistance = data.distLY * 2.5
    if (data.distLY > 100) {
      sceneDistance = 250 + Math.log10(data.distLY) * 400
    }

    const typeData = planetTypes[data.type]
    
    // Generate unique procedural texture for this planet
    const seed = index * 137.5 // Golden angle for variety
    const texture = generatePlanetTexture(data.type, seed)

    const geo = new THREE.SphereGeometry(data.size, 64, 64)
    const mat = new THREE.MeshStandardMaterial({ 
      map: texture,
      metalness: typeData.metalness, 
      roughness: typeData.roughness
    })
    
    if (typeData.emissive) {
      mat.emissive = new THREE.Color(typeData.emissiveColor)
      mat.emissiveIntensity = 0.6
    }
    
    const mesh = new THREE.Mesh(geo, mat)
    
    // Better 3D distribution around Earth
    const theta = (index * 2.4) + Math.random() * 0.5 // Golden angle distribution
    const phi = Math.acos(2 * (index / exoplanetData.length) - 1) + (Math.random() - 0.5) * 0.3
    mesh.position.set(
      sceneDistance * Math.sin(phi) * Math.cos(theta),
      (Math.random() - 0.5) * sceneDistance * 0.3,
      sceneDistance * Math.sin(phi) * Math.sin(theta)
    )

    // Calculate estimated mass based on radius (simplified)
    const estimatedMass = Math.pow(data.size / 6, 3).toFixed(2)
    
    // Determine habitability zone (simplified - based on distance and type)
    let habitable = 'No'
    if ((data.type === 'oceanic' || data.type === 'rocky') && data.size > 4 && data.size < 10) {
      habitable = 'Posible'
    }
    
    mesh.userData = {
      name: data.name,
      radius: (data.size / 6).toFixed(2) + ' R⊕',
      mass: estimatedMass + ' M⊕',
      distance: data.distLY.toFixed(0) + ' años luz',
      type: data.type.charAt(0).toUpperCase() + data.type.slice(1),
      habitable: habitable
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
    const intersects = raycaster.intersectObjects([earth, ...planets])
    
    if(intersects.length > 0 && intersects[0].distance < 100){
      const hit = intersects[0].object
      if(hovered !== hit){
        hovered = hit
        showTooltip(hit.userData)
      }
    } else {
      if(hovered){
        hovered = null
        hideTooltip()
      }
    }
  }

  function showTooltip(data){
    // Color based on habitability
    const habitableColor = data.habitable === 'Posible' ? '#00ff88' : '#999'
    
    tooltip.innerHTML = `
      <div style="font-size: 20px; font-weight: 700; margin-bottom: 10px; color: #fff; border-bottom: 1px solid rgba(77, 166, 255, 0.3); padding-bottom: 8px;">
        ${data.name}
      </div>
      <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px 16px; font-size: 13px;">
        <div style="color: rgba(255,255,255,0.6);">Tipo:</div>
        <div style="color: #4da6ff; font-weight: 600;">${data.type}</div>
        
        <div style="color: rgba(255,255,255,0.6);">Radio:</div>
        <div style="color: #fff;">${data.radius}</div>
        
        <div style="color: rgba(255,255,255,0.6);">Masa est.:</div>
        <div style="color: #fff;">${data.mass}</div>
        
        <div style="color: rgba(255,255,255,0.6);">Distancia:</div>
        <div style="color: #ff9933; font-weight: 600;">${data.distance}</div>
        
        <div style="color: rgba(255,255,255,0.6);">Habitabilidad:</div>
        <div style="color: ${habitableColor}; font-weight: 600;">${data.habitable}</div>
      </div>
    `
    tooltip.style.left = '50%'
    tooltip.style.top = '80px'
    tooltip.style.transform = 'translateX(-50%) scale(1)'
    tooltip.style.animation = 'tooltipPulse 2s ease-in-out infinite'
    tooltip.classList.add('show')
  }
  function hideTooltip(){
    tooltip.style.animation = 'none'
    tooltip.classList.remove('show')
  }

  // "Return to Earth" button
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

  // Turbo visual feedback (sin botón visible)
  const turboIndicator = document.createElement('div')
  turboIndicator.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s;
    z-index: 50;
  `
  turboIndicator.innerHTML = `
    <div style="
      font-size: 48px;
      color: #ff6600;
      text-shadow: 0 0 20px rgba(255, 102, 0, 0.8), 0 0 40px rgba(255, 102, 0, 0.4);
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 900;
      letter-spacing: 4px;
    ">⚡ TURBO ⚡</div>
  `
  document.body.appendChild(turboIndicator)
  
  function updateTurboButton() {
    if (turboMode) {
      turboIndicator.style.opacity = '0.8'
      renderer.domElement.style.filter = 'contrast(1.1) saturate(1.2)'
    } else {
      turboIndicator.style.opacity = '0'
      renderer.domElement.style.filter = 'none'
    }
  }

  // Planet search functionality
  const searchInput = document.getElementById('search-input')
  const searchResults = document.getElementById('search-results')
  
  function navigateToPlanet(planet) {
    // Move camera near the planet
    const targetPos = planet.position.clone()
    const offset = new THREE.Vector3(0, 5, planet.geometry.parameters.radius * 3)
    camera.position.copy(targetPos).add(offset)
    camera.lookAt(targetPos)
    
    // Show tooltip
    showTooltip(planet.userData)
    setTimeout(hideTooltip, 5000)
  }
  
  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim()
    searchResults.innerHTML = ''
    
    if (query.length < 2) return
    
    // Filter planets by name
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

  // Animation loop with first-person movement
  const clock = new THREE.Clock()
  function animate(){
    requestAnimationFrame(animate)
    const delta = clock.getDelta()
    
    // First-person movement (solo cuando pointer lock está activo)
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
    
    // Planets rotation
    earth.rotation.y += delta * 0.05
    planets.forEach(p => { p.rotation.y += delta * 0.03 })
    
    // Turbo speed lines effect
    if (turboMode && controls.isLocked) {
      const speed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z + velocity.y * velocity.y)
      if (speed > 10) {
        renderer.domElement.style.filter = 'contrast(1.15) saturate(1.3) brightness(1.05)'
      }
    } else {
      renderer.domElement.style.filter = 'none'
    }
    
    // Starfield rotation for depth
    milkyWayPoints.rotation.y += delta * 0.0001
    mediumStarPoints.rotation.y -= delta * 0.0002
    brightStarPoints.rotation.x += delta * 0.0001
    
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
  instructions.style.background = 'rgba(0,0,0,0.85)'
  instructions.style.color = 'white'
  instructions.style.borderRadius = '8px'
  instructions.style.fontFamily = 'Space Grotesk, sans-serif'
  instructions.style.fontSize = '14px'
  instructions.style.textAlign = 'center'
  instructions.style.zIndex = '100'
  instructions.style.backdropFilter = 'blur(8px)'
  instructions.style.border = '1px solid rgba(77, 166, 255, 0.3)'
  instructions.innerHTML = 'Haz clic para empezar • WASD: Movimiento • Espacio/Ctrl: Subir/Bajar • Shift: TURBO • Ratón: Mirar alrededor'
  document.body.appendChild(instructions)
  
  controls.addEventListener('lock', () => {
    instructions.style.display = 'none'
  })
  controls.addEventListener('unlock', () => {
    instructions.style.display = 'block'
  })

})()
