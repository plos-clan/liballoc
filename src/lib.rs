#![no_std]
#![no_main]

use core::alloc::Layout;
use core::ffi::c_void;
use core::{panic::PanicInfo, ptr::NonNull};
use talc::{ClaimOnOom, Span, Talc, Talck};

#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

static ALLOCATOR: Talck<spin::Mutex<()>, ClaimOnOom> =
    Talc::new(unsafe { ClaimOnOom::new(Span::empty()) }).lock();

#[no_mangle]
pub unsafe extern "C" fn heap_init(address: *mut u8, size: usize) -> bool {
    let arena = Span::from_base_size(address as *mut u8, size);
    ALLOCATOR.lock().claim(arena).is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    let size_with_meta = size + size_of::<usize>();
    let layout = Layout::from_size_align_unchecked(size_with_meta, align_of::<usize>());

    match ALLOCATOR.lock().malloc(layout) {
        Ok(ptr) => {
            let size_ptr = ptr.as_ptr() as *mut usize;
            size_ptr.write(size);
            (size_ptr.offset(1) as *mut u8) as *mut c_void
        }
        Err(_) => return core::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let size_ptr = ptr as *mut usize;
    let origin_ptr = size_ptr.offset(-1);

    let size_with_meta = origin_ptr.read() + size_of::<usize>();
    let layout = Layout::from_size_align_unchecked(size_with_meta, align_of::<usize>());

    let origin_ptr = NonNull::new_unchecked(origin_ptr as *mut u8);
    ALLOCATOR.lock().free(origin_ptr, layout);
}
