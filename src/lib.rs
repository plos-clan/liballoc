#![no_std]
#![no_main]

use core::alloc::Layout;
use core::ffi::c_void;
use core::{panic::PanicInfo, ptr::NonNull, ptr};
use talc::{ClaimOnOom, Span, Talc, Talck};
use core::cmp; // Import cmp for min

#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

static ALLOCATOR: Talck<spin::Mutex<()>, ClaimOnOom> =
    Talc::new(unsafe { ClaimOnOom::new(Span::empty()) }).lock();

#[no_mangle]
pub unsafe extern "C" fn heap_init(address: *mut u8, size: usize) -> bool {
    let arena = Span::from_base_size(address, size);
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
        Err(_) => ptr::null_mut(),
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

#[no_mangle]
pub unsafe extern "C" fn get_allocator_size(ptr: *mut c_void) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let user_ptr_as_usize = ptr as *mut usize;
    let metadata_ptr = user_ptr_as_usize.sub(1);
    let size = metadata_ptr.read();
    size
}

#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    if ptr.is_null() {
        return malloc(new_size);
    }

    if new_size == 0 {
        free(ptr);
        return ptr::null_mut();
    }

    let old_size = get_allocator_size(ptr);
    if new_size == old_size {
        return ptr;
    }

    let new_ptr = malloc(new_size);

    if new_ptr.is_null() {
        return ptr::null_mut();
    }
    let copy_size = cmp::min(old_size, new_size);
    ptr::copy_nonoverlapping(ptr as *const u8, new_ptr as *mut u8, copy_size);
    free(ptr);
    new_ptr
}