
namespace VertoTest.Models
{
    public class ImageViewModel
    {
        public string? AltText { get; set; }

        public int? StaticContentID { get; set; }

        public required IFormFile ImgFile { get; set; }

    }
}
